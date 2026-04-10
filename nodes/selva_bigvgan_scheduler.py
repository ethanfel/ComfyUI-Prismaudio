"""SelVA BigVGAN Vocoder Scheduler — runs a sweep of vocoder fine-tuning experiments.

Each experiment inherits from a shared `base` config and overrides specific keys.
Audio clips are loaded once and reused across all experiments. Results are written
to `experiment_summary.json` (updated after each completed run) and a comparison
loss-curve image.

JSON format:
    {
      "name": "bigvgan_sweep",
      "description": "optional note",
      "data_dir": "/path/to/audio/clips",
      "output_root": "/path/to/output",
      "base": { "train_mode": "snake_alpha_only", "steps": 2000, "lr": 1e-4, ... },
      "experiments": [
        {"id": "baseline", "description": "..."},
        {"id": "all_5k", "train_mode": "all_params", "steps": 5000, "lr": 1e-5},
        ...
      ]
    }
"""

import copy
import csv
import json
import threading
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torchaudio

import comfy.utils
import comfy.model_management
import folder_paths

from .utils import SELVA_CATEGORY, get_device, soft_empty_cache
from .selva_bigvgan_trainer import (
    _do_train,
    _pregenerate_lora_mels,
    _load_wav,
)
from .selva_lora_trainer import _smooth_losses, _pil_to_tensor
from .selva_lora_scheduler import (
    _get_system_info,
    _resolve_path,
    _draw_comparison_curves,
)


# Defaults mirror SelvaBigvganTrainer INPUT_TYPES defaults
_PARAM_DEFAULTS = {
    "train_mode":           "snake_alpha_only",
    "steps":                2000,
    "lr":                   1e-4,
    "batch_size":           4,
    "segment_seconds":      2.0,
    "lambda_l2sp":          1e-3,
    "use_gafilter":         True,
    "gafilter_kernel_size": 9,
    "lambda_phase":         1.0,
    "save_every":           500,
    "seed":                 42,
    "discriminator_path":   "",
    "lora_adapter":         "",
}


def _merge_config(base: dict, experiment: dict) -> dict:
    """Merge param defaults + file base + experiment overrides."""
    cfg = dict(_PARAM_DEFAULTS)
    cfg.update(base)
    cfg.update({k: v for k, v in experiment.items() if k not in ("id", "description")})
    return cfg


def _parse_training_log(log_path: Path) -> list:
    """Parse BigVGAN training CSV → list of total_loss values."""
    losses = []
    if not log_path.exists():
        return losses
    try:
        with open(log_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                losses.append(float(row["total_loss"]))
    except Exception:
        pass
    return losses


def _loss_at_steps(loss_history: list, log_interval: int, save_every: int,
                   total_steps: int) -> dict:
    """Build {step: loss} at each save_every boundary.

    Uses round-to-nearest to handle log_interval that doesn't divide
    save_every evenly (e.g. steps=3000 → log_interval=150, save_every=1000).
    """
    result = {}
    for target in range(save_every, total_steps + 1, save_every):
        # loss_history[i] = loss at step (i+1)*log_interval
        idx = round(target / log_interval) - 1
        if 0 <= idx < len(loss_history):
            result[str(target)] = round(loss_history[idx], 6)
    return result


class SelvaBigvganScheduler:
    """Runs a sweep of BigVGAN vocoder fine-tuning experiments from a JSON file.

    Audio clips are loaded once and reused across all experiments. Each experiment
    deep-copies the vocoder and trains independently. Results are written to
    `experiment_summary.json` after every completed run so partial results are
    preserved if the sweep is interrupted.
    """

    OUTPUT_NODE  = True
    CATEGORY     = SELVA_CATEGORY
    FUNCTION     = "run"
    RETURN_TYPES  = ("STRING", "IMAGE")
    RETURN_NAMES  = ("summary_path", "comparison_curves")
    OUTPUT_TOOLTIPS = (
        "Path to experiment_summary.json — share this file to compare runs.",
        "All smoothed loss curves overlaid on the same axes.",
    )
    DESCRIPTION = (
        "Runs a series of BigVGAN vocoder fine-tuning experiments defined in a JSON sweep file. "
        "Audio clips are loaded once and reused across all experiments. "
        "Results (loss, config, checkpoint paths) are collected in experiment_summary.json."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("SELVA_MODEL",),
                "experiments_file": ("STRING", {
                    "default": "bigvgan_experiments.json",
                    "tooltip": (
                        "Path to JSON sweep file. Relative paths resolve to the ComfyUI "
                        "models directory; absolute paths are used as-is."
                    ),
                }),
            }
        }

    def run(self, model, experiments_file):
        # ------------------------------------------------------------------
        # 1. Read + validate the JSON file
        # ------------------------------------------------------------------
        exp_path = Path(experiments_file.strip())
        if not exp_path.is_absolute():
            candidate = Path(folder_paths.models_dir) / exp_path
            if not candidate.exists():
                candidate = Path(folder_paths.get_output_directory()) / exp_path
            exp_path = candidate
        if not exp_path.exists():
            raise FileNotFoundError(
                f"[BigVGAN Scheduler] Experiment file not found: {exp_path}"
            )

        spec = json.loads(exp_path.read_text(encoding="utf-8"))

        if "experiments" not in spec or not spec["experiments"]:
            raise ValueError("[BigVGAN Scheduler] 'experiments' list is missing or empty.")
        for i, exp in enumerate(spec["experiments"]):
            if "id" not in exp:
                raise ValueError(
                    f"[BigVGAN Scheduler] Experiment at index {i} is missing required 'id' field."
                )

        sweep_name  = spec.get("name", exp_path.stem)
        description = spec.get("description", "")
        base_cfg    = spec.get("base", {})

        # ------------------------------------------------------------------
        # 2. Resolve data_dir and output_root
        # ------------------------------------------------------------------
        if "data_dir" not in spec:
            raise ValueError("[BigVGAN Scheduler] 'data_dir' is required in the sweep file.")
        data_dir    = _resolve_path(spec["data_dir"])
        output_root = _resolve_path(spec.get("output_root", f"bigvgan_sweeps/{sweep_name}"))
        output_root.mkdir(parents=True, exist_ok=True)

        device = get_device()
        mode   = model["mode"]
        dtype  = model["dtype"]
        feature_utils = model["feature_utils"]
        mel_converter = feature_utils.mel_converter
        strategy      = model["strategy"]

        if mode == "16k":
            original_vocoder = feature_utils.tod.vocoder.vocoder
            sample_rate = 16_000
        elif mode == "44k":
            original_vocoder = feature_utils.tod.vocoder
            sample_rate = 44_100
        else:
            raise ValueError(f"[BigVGAN Scheduler] Unknown mode: {mode}")

        print(f"\n[BigVGAN Scheduler] Sweep '{sweep_name}': "
              f"{len(spec['experiments'])} experiment(s)", flush=True)
        if description:
            print(f"[BigVGAN Scheduler] {description}", flush=True)
        print(f"[BigVGAN Scheduler] data_dir    = {data_dir}", flush=True)
        print(f"[BigVGAN Scheduler] output_root = {output_root}\n", flush=True)

        # ------------------------------------------------------------------
        # 3. Load audio clips once
        # ------------------------------------------------------------------
        # Find minimum segment length across all experiments so we load enough
        min_segment_seconds = float("inf")
        for exp in spec["experiments"]:
            cfg = _merge_config(base_cfg, exp)
            min_segment_seconds = min(min_segment_seconds, float(cfg.get("segment_seconds", 2.0)))
        min_segment_samples = int(min_segment_seconds * sample_rate)

        audio_files = []
        for ext in ("*.wav", "*.flac", "*.mp3", "*.ogg", "*.aac"):
            audio_files.extend(data_dir.rglob(ext))
        if not audio_files:
            raise FileNotFoundError(f"[BigVGAN Scheduler] No audio files in {data_dir}")

        print(f"[BigVGAN Scheduler] Loading {len(audio_files)} audio files...", flush=True)
        clips = []
        for af in audio_files:
            try:
                wav, sr = _load_wav(af)
                if wav.shape[0] > 1:
                    wav = wav.mean(0, keepdim=True)
                if sr != sample_rate:
                    wav = torchaudio.functional.resample(wav, sr, sample_rate)
                wav = wav.squeeze(0)  # [L]
                if wav.shape[0] >= min_segment_samples:
                    clips.append(wav.cpu())
                else:
                    print(f"  [BigVGAN Scheduler] Skip {af.name}: "
                          f"shorter than {min_segment_seconds}s", flush=True)
            except Exception as e:
                print(f"  [BigVGAN Scheduler] Failed {af.name}: {e}", flush=True)

        if not clips:
            raise RuntimeError(
                f"[BigVGAN Scheduler] No usable clips (need audio >= {min_segment_seconds}s)"
            )
        print(f"[BigVGAN Scheduler] {len(clips)} clips ready\n", flush=True)

        # ------------------------------------------------------------------
        # 4. Offload unused components to free VRAM
        # ------------------------------------------------------------------
        comfy.model_management.unload_all_models()
        feature_utils.to("cpu")
        if "generator" in model:
            model["generator"].to("cpu")
        if "video_enc" in model:
            model["video_enc"].to("cpu")
        soft_empty_cache()

        # ------------------------------------------------------------------
        # 5. Pre-compute text CLIP embeddings if any experiment uses LoRA
        # ------------------------------------------------------------------
        text_clip_cache = {}
        any_lora = any(
            _merge_config(base_cfg, exp).get("lora_adapter", "")
            for exp in spec["experiments"]
        )
        if any_lora:
            npz_files = sorted(data_dir.glob("*.npz"))
            if npz_files:
                prompt_map = {}
                prompts_file = data_dir / "prompts.txt"
                if prompts_file.exists():
                    for line in prompts_file.read_text(encoding="utf-8").splitlines():
                        line = line.strip()
                        if not line or line.startswith("#"):
                            continue
                        if "|" in line:
                            fname, prompt = line.split("|", 1)
                            prompt_map[fname.strip()] = prompt.strip()
                default_prompt = data_dir.name

                clip_model = feature_utils.clip_model
                if clip_model is not None:
                    clip_model.to(device)
                try:
                    for npz_path in npz_files:
                        data = dict(np.load(str(npz_path), allow_pickle=False))
                        prompt = prompt_map.get(
                            npz_path.name, data.get("prompt", default_prompt)
                        )
                        if isinstance(prompt, np.ndarray):
                            prompt = str(prompt)
                        tc = feature_utils.encode_text_clip([prompt])
                        text_clip_cache[npz_path.name] = tc.clone().detach().cpu()
                finally:
                    if clip_model is not None:
                        clip_model.to("cpu")
                    soft_empty_cache()
                    if device.type == "cuda":
                        torch.cuda.empty_cache()
                print(f"[BigVGAN Scheduler] Pre-encoded {len(text_clip_cache)} "
                      f"CLIP embeddings", flush=True)

        # ------------------------------------------------------------------
        # 6. Build or restore the summary (resume-aware)
        # ------------------------------------------------------------------
        summary_path  = output_root / "experiment_summary.json"
        completed_ids = set()
        all_curve_data = []

        if summary_path.exists():
            try:
                existing = json.loads(summary_path.read_text(encoding="utf-8"))
                for rec in existing.get("experiments", []):
                    if rec.get("results", {}).get("status") == "completed":
                        completed_ids.add(rec["id"])
                        lh = rec["results"].get("loss_history", [])
                        all_curve_data.append({
                            "id":           rec["id"],
                            "loss_history": lh,
                            "log_interval": rec["results"].get("log_interval", 100),
                            "start_step":   0,
                        })
                summary = existing
                summary["completed_at"] = None
                if completed_ids:
                    print(f"[BigVGAN Scheduler] Resuming — skipping "
                          f"{len(completed_ids)} completed: "
                          f"{sorted(completed_ids)}", flush=True)
            except Exception as e:
                print(f"[BigVGAN Scheduler] Could not read existing summary "
                      f"({e}) — starting fresh", flush=True)
                completed_ids = set()
                all_curve_data = []
                summary = None

        if not completed_ids:
            summary = {
                "sweep_name":   sweep_name,
                "description":  description,
                "sweep_file":   str(exp_path),
                "started_at":   datetime.now(timezone.utc).isoformat(),
                "completed_at": None,
                "system":       _get_system_info(),
                "data_dir":     str(data_dir),
                "n_clips":      len(clips),
                "experiments":  [],
            }

        def _write_summary():
            summary_path.write_text(
                json.dumps(summary, indent=2), encoding="utf-8"
            )

        _write_summary()

        # ------------------------------------------------------------------
        # 7. Compute total steps for progress bar
        # ------------------------------------------------------------------
        total_steps = 0
        for exp in spec["experiments"]:
            if exp["id"] not in completed_ids:
                cfg = _merge_config(base_cfg, exp)
                total_steps += int(cfg.get("steps", 2000))
        pbar = comfy.utils.ProgressBar(max(total_steps, 1))

        # ------------------------------------------------------------------
        # 8. Run experiments in a worker thread
        # ------------------------------------------------------------------
        # BigVGAN training requires a fresh thread because ComfyUI runs nodes
        # inside torch.inference_mode(). inference_mode is thread-local — a
        # new thread starts with it OFF, so all tensor operations produce
        # normal autograd-compatible tensors.
        _exc = [None]

        def _worker():
            try:
                for exp in spec["experiments"]:
                    exp_id   = exp["id"]
                    exp_desc = exp.get("description", "")

                    if exp_id in completed_ids:
                        print(f"[BigVGAN Scheduler] Skipping '{exp_id}' "
                              f"(already completed)", flush=True)
                        continue

                    cfg = _merge_config(base_cfg, exp)

                    # ── Extract experiment parameters ────────────────────
                    train_mode   = str(cfg.get("train_mode", "snake_alpha_only"))
                    exp_steps    = int(cfg.get("steps", 2000))
                    exp_lr       = float(cfg.get("lr", 1e-4))
                    exp_bs       = int(cfg.get("batch_size", 4))
                    exp_seg_s    = float(cfg.get("segment_seconds", 2.0))
                    exp_l2sp     = float(cfg.get("lambda_l2sp", 1e-3))
                    exp_gafilter = bool(cfg.get("use_gafilter", True))
                    exp_gaf_ks   = int(cfg.get("gafilter_kernel_size", 9))
                    exp_phase    = float(cfg.get("lambda_phase", 1.0))
                    exp_save     = int(cfg.get("save_every", 500))
                    exp_seed     = int(cfg.get("seed", 42))
                    exp_disc     = str(cfg.get("discriminator_path", ""))
                    exp_lora     = str(cfg.get("lora_adapter", ""))

                    segment_samples = int(exp_seg_s * sample_rate)

                    # Filter clips long enough for this experiment
                    exp_clips = [c for c in clips if c.shape[0] >= segment_samples]
                    if not exp_clips:
                        print(f"[BigVGAN Scheduler] '{exp_id}' skipped: "
                              f"no clips >= {exp_seg_s}s", flush=True)
                        summary["experiments"].append({
                            "id": exp_id, "description": exp_desc,
                            "config": dict(cfg),
                            "results": {
                                "status": "failed",
                                "error": f"No clips >= {exp_seg_s}s",
                                "duration_seconds": 0,
                            },
                            "checkpoint_path": None,
                            "output_dir": str(output_root / exp_id),
                        })
                        _write_summary()
                        continue

                    # ── Resolve discriminator path ───────────────────────
                    disc_path = None
                    if exp_disc:
                        disc_path = Path(exp_disc.strip())
                        if not disc_path.is_absolute():
                            disc_path = (
                                Path(folder_paths.get_output_directory()) / disc_path
                            )
                        if not disc_path.exists():
                            print(f"[BigVGAN Scheduler] '{exp_id}': "
                                  f"discriminator not found: {disc_path}",
                                  flush=True)
                            disc_path = None

                    # ── Pre-generate LoRA mels (disk-cached) ─────────────
                    lora_mel_pairs = None
                    if exp_lora:
                        lora_path = Path(exp_lora.strip())
                        if not lora_path.is_absolute():
                            lora_path = Path(folder_paths.base_path) / lora_path
                        if lora_path.exists():
                            seq_cfg = model["seq_cfg"]
                            lora_mel_pairs = _pregenerate_lora_mels(
                                model, data_dir, str(lora_path),
                                device, dtype, sample_rate,
                                seq_cfg.duration, seed=exp_seed,
                                cache_dir=str(output_root),
                                text_clip_cache=text_clip_cache,
                            )
                            if not lora_mel_pairs:
                                print(f"[BigVGAN Scheduler] '{exp_id}': "
                                      f"no LoRA mel pairs generated",
                                      flush=True)
                                lora_mel_pairs = None
                            if device.type == "cuda":
                                torch.cuda.empty_cache()
                        else:
                            print(f"[BigVGAN Scheduler] '{exp_id}': "
                                  f"LoRA adapter not found: {lora_path}",
                                  flush=True)

                    # ── Output dir ───────────────────────────────────────
                    exp_dir = output_root / exp_id
                    exp_dir.mkdir(parents=True, exist_ok=True)
                    out_path = exp_dir / f"bigvgan_{exp_id}.pt"

                    print(f"\n[BigVGAN Scheduler] ── Experiment '{exp_id}' ──",
                          flush=True)
                    if exp_desc:
                        print(f"[BigVGAN Scheduler] {exp_desc}", flush=True)
                    print(f"[BigVGAN Scheduler] mode={train_mode}  "
                          f"steps={exp_steps}  lr={exp_lr}  bs={exp_bs}  "
                          f"seg={exp_seg_s}s  gafilter={exp_gafilter}  "
                          f"phase={exp_phase}  l2sp={exp_l2sp}", flush=True)

                    exp_record = {
                        "id":          exp_id,
                        "description": exp_desc,
                        "config": {
                            "train_mode": train_mode, "steps": exp_steps,
                            "lr": exp_lr, "batch_size": exp_bs,
                            "segment_seconds": exp_seg_s,
                            "lambda_l2sp": exp_l2sp,
                            "use_gafilter": exp_gafilter,
                            "gafilter_kernel_size": exp_gaf_ks,
                            "lambda_phase": exp_phase,
                            "save_every": exp_save, "seed": exp_seed,
                            "discriminator_path": exp_disc,
                            "lora_adapter": exp_lora,
                        },
                        "results":         {"status": "running"},
                        "checkpoint_path": None,
                        "output_dir":      str(exp_dir),
                    }
                    summary["experiments"].append(exp_record)
                    _write_summary()

                    t_start = time.monotonic()
                    try:
                        # Ensure mel_converter is on device for this experiment
                        mel_converter.to(device)

                        # Fresh vocoder copy — _do_train modifies it in-place
                        vocoder_copy = copy.deepcopy(original_vocoder)

                        checkpoint_path = _do_train(
                            vocoder_copy, mel_converter, exp_clips,
                            device, dtype, strategy, feature_utils,
                            segment_samples, sample_rate,
                            train_mode, exp_steps, exp_lr, exp_bs,
                            exp_l2sp, exp_gafilter, exp_gaf_ks,
                            exp_phase, exp_save, exp_seed,
                            out_path, disc_path, pbar,
                            lora_mel_pairs,
                        )

                        duration = time.monotonic() - t_start

                        # Parse training CSV for loss history
                        log_path = exp_dir / f"bigvgan_{exp_id}_training_log.csv"
                        loss_history = _parse_training_log(log_path)
                        log_interval = max(1, exp_steps // 20)
                        smoothed = (
                            _smooth_losses(loss_history)
                            if loss_history else []
                        )

                        final_loss = (
                            round(smoothed[-1], 6) if smoothed else None
                        )
                        min_loss = (
                            round(min(smoothed), 6) if smoothed else None
                        )
                        min_idx = (
                            smoothed.index(min(smoothed))
                            if smoothed else None
                        )
                        min_loss_step = (
                            (min_idx + 1) * log_interval
                            if min_idx is not None else None
                        )

                        if loss_history:
                            quarter = max(1, len(loss_history) // 4)
                            loss_std = round(
                                float(np.std(loss_history[-quarter:])), 6
                            )
                        else:
                            loss_std = None

                        exp_record["results"] = {
                            "status":               "completed",
                            "final_loss":           final_loss,
                            "min_loss":             min_loss,
                            "min_loss_step":        min_loss_step,
                            "loss_std_last_quarter": loss_std,
                            "loss_at_steps":        _loss_at_steps(
                                loss_history, log_interval,
                                exp_save, exp_steps,
                            ),
                            "loss_history": [
                                round(v, 6) for v in loss_history
                            ],
                            "log_interval":     log_interval,
                            "duration_seconds": round(duration, 1),
                        }
                        exp_record["checkpoint_path"] = checkpoint_path

                        all_curve_data.append({
                            "id":           exp_id,
                            "loss_history": loss_history,
                            "log_interval": log_interval,
                            "start_step":   0,
                        })

                    except Exception as e:
                        duration = time.monotonic() - t_start
                        print(f"[BigVGAN Scheduler] Experiment '{exp_id}' "
                              f"failed: {e}", flush=True)
                        traceback.print_exc()
                        exp_record["results"] = {
                            "status":           "failed",
                            "error":            str(e),
                            "duration_seconds": round(duration, 1),
                        }
                    finally:
                        # Clean up vocoder copy to free VRAM
                        soft_empty_cache()

                    _write_summary()

            except Exception as e:
                _exc[0] = e
                traceback.print_exc()

        t = threading.Thread(target=_worker, daemon=True)
        t.start()
        t.join()

        if _exc[0] is not None:
            raise _exc[0]

        # ------------------------------------------------------------------
        # 9. Finalise summary
        # ------------------------------------------------------------------
        summary["completed_at"] = datetime.now(timezone.utc).isoformat()
        _write_summary()
        print(f"\n[BigVGAN Scheduler] Sweep complete. "
              f"Summary: {summary_path}", flush=True)

        # ------------------------------------------------------------------
        # 10. Comparison image
        # ------------------------------------------------------------------
        comparison_img = _draw_comparison_curves(all_curve_data)
        comparison_img.save(str(output_root / "loss_comparison.png"))
        comparison_tensor = _pil_to_tensor(comparison_img)

        return (str(summary_path), comparison_tensor)
