"""SelVA LoRA Evaluator — generates eval samples from multiple adapters for comparison.

JSON format:
    {
      "name": "eval_batch_1",
      "data_dir": "/path/to/features",
      "output_dir": "/path/to/evals/batch1",
      "steps": 25,
      "seed": 42,
      "adapters": [
        {"id": "baseline"},
        {"id": "lr_3e4_10k", "path": "/path/to/adapter_final.pt"},
        {"id": "lr_5e4_10k", "path": "/path/to/adapter_final.pt"}
      ]
    }

Empty / missing "path" = baseline (no LoRA applied).
"""

import copy
import json
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torchaudio

import comfy.utils
import folder_paths

from .utils import SELVA_CATEGORY, get_device, soft_empty_cache
from .selva_lora_trainer import (
    _prepare_dataset,
    _eval_sample,
    _spectral_metrics,
    _save_spectrogram,
    _pil_to_tensor,
    _find_audio,
    _load_audio,
)
from selva_core.model.lora import apply_lora, load_lora


def _resolve_path(raw: str) -> Path:
    p = Path(raw.strip())
    unix_style_on_windows = sys.platform == "win32" and p.is_absolute() and not p.drive
    if not p.is_absolute() or unix_style_on_windows:
        p = Path(folder_paths.get_output_directory()) / p.relative_to(p.anchor)
    return p


def _safe_stem(adapter_id: str) -> str:
    """Replace characters illegal in filenames."""
    for ch in r'/\:*?"<>|':
        adapter_id = adapter_id.replace(ch, "_")
    return adapter_id


def _draw_metric_comparison(adapter_ids: list, metrics_list: list, output_path: Path):
    """Draw a 2×2 grid of horizontal bar charts comparing spectral metrics.

    Saves a PNG to output_path and returns a ComfyUI IMAGE tensor.
    """
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg

    METRICS = [
        ("hf_energy_ratio",      "HF Energy Ratio (>4 kHz)"),
        ("spectral_centroid_hz", "Spectral Centroid (Hz)"),
        ("spectral_flatness",    "Spectral Flatness"),
        ("temporal_variance",    "Temporal Variance"),
    ]
    COLORS = [
        "#4285F4", "#EA4335", "#34A853", "#FBBC05",
        "#9B59B6", "#1ABC9C", "#E67E22", "#95A5A6",
    ]

    fig = Figure(figsize=(12, max(4, len(adapter_ids) * 0.6 + 2)), dpi=110, tight_layout=True)
    axes = [fig.add_subplot(2, 2, i + 1) for i in range(4)]

    for ax, (key, title) in zip(axes, METRICS):
        values = []
        colors = []
        for i, m in enumerate(metrics_list):
            v = m.get(key, 0.0) if m else 0.0
            values.append(v)
            colors.append(COLORS[i % len(COLORS)])

        bars = ax.barh(adapter_ids, values, color=colors, height=0.6)
        ax.set_title(title, fontsize=9)
        ax.set_xlabel(key, fontsize=8)
        ax.tick_params(axis="y", labelsize=7)
        ax.tick_params(axis="x", labelsize=7)

        # Value labels on bars
        for bar, val in zip(bars, values):
            w = bar.get_width()
            ax.text(w * 1.01, bar.get_y() + bar.get_height() / 2,
                    f"{val:.3f}", va="center", ha="left", fontsize=6)

    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    canvas.print_figure(str(output_path), dpi=110)

    buf = canvas.buffer_rgba()
    w, h = canvas.get_width_height()
    arr = np.frombuffer(buf, dtype=np.uint8).reshape(h, w, 4)[:, :, :3]
    from PIL import Image
    return _pil_to_tensor(Image.fromarray(arr))


class SelvaLoraEvaluator:
    """Evaluates a batch of LoRA adapters on a fixed reference clip.

    Generates one audio sample per adapter, computes spectral metrics for each,
    and produces a comparison chart. Use this after a sweep to compare candidates
    before running the next round of training.
    """

    OUTPUT_NODE = True
    CATEGORY    = SELVA_CATEGORY
    FUNCTION    = "run"
    RETURN_TYPES  = ("STRING", "IMAGE")
    RETURN_NAMES  = ("summary_path", "comparison_image")
    OUTPUT_TOOLTIPS = (
        "Path to eval_summary.json — contains spectral metrics per adapter.",
        "Bar chart comparing spectral metrics across all evaluated adapters.",
    )
    DESCRIPTION = (
        "Evaluates multiple LoRA adapters by generating one audio sample per adapter "
        "from a fixed reference clip, then collects spectral metrics for comparison. "
        "Input is a JSON file listing adapter paths. Empty path = baseline (no LoRA)."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("SELVA_MODEL",),
                "eval_file": ("STRING", {
                    "default": "eval_batch.json",
                    "tooltip": (
                        "Path to the JSON evaluation spec. Relative paths resolve "
                        "to the ComfyUI output directory. "
                        "Each adapter entry needs an 'id' and an optional 'path'. "
                        "Omit 'path' for a no-LoRA baseline."
                    ),
                }),
            }
        }

    def run(self, model, eval_file):
        # ------------------------------------------------------------------
        # 1. Resolve and parse the JSON file
        # ------------------------------------------------------------------
        eval_path = Path(eval_file.strip())
        if not eval_path.is_absolute():
            candidate = Path(folder_paths.models_dir) / eval_path
            if not candidate.exists():
                candidate = Path(folder_paths.get_output_directory()) / eval_path
            eval_path = candidate
        if not eval_path.exists():
            raise FileNotFoundError(f"[LoRA Evaluator] Eval file not found: {eval_path}")

        spec = json.loads(eval_path.read_text(encoding="utf-8"))

        if "adapters" not in spec or not spec["adapters"]:
            raise ValueError("[LoRA Evaluator] 'adapters' list is missing or empty.")
        for i, a in enumerate(spec["adapters"]):
            if "id" not in a:
                raise ValueError(f"[LoRA Evaluator] Adapter at index {i} missing 'id'.")

        if "data_dir" not in spec:
            raise ValueError("[LoRA Evaluator] 'data_dir' is required.")
        if "output_dir" not in spec:
            raise ValueError("[LoRA Evaluator] 'output_dir' is required.")

        name       = spec.get("name", eval_path.stem)
        data_dir   = _resolve_path(spec["data_dir"])
        output_dir = _resolve_path(spec["output_dir"])
        steps      = int(spec.get("steps", 25))
        seed       = int(spec.get("seed",  42))
        output_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n[LoRA Evaluator] '{name}': {len(spec['adapters'])} adapter(s)", flush=True)
        print(f"[LoRA Evaluator] data_dir   = {data_dir}", flush=True)
        print(f"[LoRA Evaluator] output_dir = {output_dir}\n", flush=True)

        # ------------------------------------------------------------------
        # 2. Prepare dataset (VAE encode once)
        # ------------------------------------------------------------------
        device = get_device()
        dtype  = model["dtype"]
        dataset = _prepare_dataset(model, data_dir, device)

        feature_utils_orig = model["feature_utils"]
        seq_cfg            = model["seq_cfg"]

        # ------------------------------------------------------------------
        # 3. Load reference audio (first clip, same one used for eval samples)
        # ------------------------------------------------------------------
        first_npz   = sorted(data_dir.glob("*.npz"))[0]
        audio_path  = _find_audio(first_npz)
        ref_record  = {"id": "reference", "path": str(audio_path) if audio_path else None,
                       "wav_path": None, "spectrogram_path": None,
                       "spectral_metrics": None, "status": "failed"}
        if audio_path:
            try:
                ref_wav = _load_audio(audio_path, seq_cfg.sampling_rate, seq_cfg.duration)
                ref_wav = ref_wav.unsqueeze(0)   # [1, L]
                import shutil
                ref_out = output_dir / f"reference{audio_path.suffix}"
                shutil.copy2(str(audio_path), str(ref_out))
                ref_record["wav_path"]         = str(ref_out)
                ref_record["spectral_metrics"] = _spectral_metrics(ref_wav, seq_cfg.sampling_rate)
                _save_spectrogram(ref_wav, seq_cfg.sampling_rate, output_dir / "reference")
                ref_record["spectrogram_path"] = str((output_dir / "reference").with_suffix(".png"))
                ref_record["status"] = "completed"
                print(f"[LoRA Evaluator] Reference: {audio_path.name}", flush=True)
            except Exception as e:
                print(f"[LoRA Evaluator] Reference load failed: {e}", flush=True)

        # ------------------------------------------------------------------
        # 4. Build summary skeleton
        # ------------------------------------------------------------------
        summary = {
            "name":         name,
            "started_at":   datetime.now(timezone.utc).isoformat(),
            "completed_at": None,
            "data_dir":     str(data_dir),
            "output_dir":   str(output_dir),
            "reference":    first_npz.name,
            "steps":        steps,
            "seed":         seed,
            "adapters":     [ref_record],
        }
        summary_path = output_dir / "eval_summary.json"

        def _write_summary():
            summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

        _write_summary()

        # ------------------------------------------------------------------
        # 5. Per-adapter evaluation loop
        # ------------------------------------------------------------------
        pbar = comfy.utils.ProgressBar(len(spec["adapters"]))

        for adapter_spec in spec["adapters"]:
            adapter_id   = adapter_spec["id"]
            adapter_path = (adapter_spec.get("path") or "").strip()
            safe_id      = _safe_stem(adapter_id)

            record = {
                "id":                adapter_id,
                "path":              adapter_path or None,
                "meta":              None,
                "wav_path":          None,
                "spectrogram_path":  None,
                "spectral_metrics":  None,
                "status":            "running",
            }

            print(f"[LoRA Evaluator] ── '{adapter_id}' ──", flush=True)

            try:
                with torch.inference_mode(False):
                    # 4a. Deep-copy generator
                    generator = copy.deepcopy(model["generator"])

                    # 4b. Apply + load LoRA if path given
                    if adapter_path:
                        pt_path = Path(adapter_path)
                        if not pt_path.is_absolute():
                            pt_path = Path(folder_paths.base_path) / pt_path
                        if not pt_path.exists():
                            raise FileNotFoundError(f"Adapter not found: {pt_path}")

                        ckpt = torch.load(str(pt_path), map_location="cpu",
                                          weights_only=False)
                        if isinstance(ckpt, dict) and "state_dict" in ckpt:
                            state_dict = ckpt["state_dict"]
                            meta       = ckpt.get("meta", {})
                        else:
                            state_dict = ckpt
                            meta       = {}

                        rank    = int(meta.get("rank",   16))
                        alpha   = float(meta.get("alpha", float(rank)))
                        target  = list(meta.get("target", ["attn.qkv"]))
                        dropout = float(meta.get("lora_dropout", 0.0))
                        record["meta"] = {"rank": rank, "alpha": alpha, "target": target}

                        n = apply_lora(generator, rank=rank, alpha=alpha,
                                       target_suffixes=tuple(target), dropout=dropout)
                        if n == 0:
                            raise RuntimeError(
                                f"apply_lora matched 0 layers (target={target})"
                            )
                        load_lora(generator, state_dict)
                        print(f"[LoRA Evaluator] Loaded {pt_path.name} "
                              f"(rank={rank}, {n} layers)", flush=True)
                    else:
                        print("[LoRA Evaluator] Baseline (no LoRA)", flush=True)

                    # 4c. Move to device and set sequence lengths
                    generator = generator.to(device, dtype)
                    generator.update_seq_lengths(
                        latent_seq_len=seq_cfg.latent_seq_len,
                        clip_seq_len=seq_cfg.clip_seq_len,
                        sync_seq_len=seq_cfg.sync_seq_len,
                    )

                    # 4d. Run inference
                    wav, sr = _eval_sample(
                        generator, feature_utils_orig, dataset,
                        seq_cfg, device, dtype,
                        num_steps=steps, seed=seed,
                    )

                if wav is None:
                    raise RuntimeError("_eval_sample returned None")

                # 4e. Save wav
                wav_path = output_dir / f"{safe_id}.wav"
                try:
                    torchaudio.save(str(wav_path), wav, sr)
                except RuntimeError:
                    import soundfile as sf
                    sf.write(str(wav_path), wav.squeeze(0).numpy(), sr)
                record["wav_path"] = str(wav_path)
                print(f"[LoRA Evaluator] Saved {wav_path.name}", flush=True)

                # 4f. Spectral metrics
                metrics = _spectral_metrics(wav, sr)
                record["spectral_metrics"] = metrics
                print(f"[LoRA Evaluator] hf={metrics['hf_energy_ratio']:.3f}  "
                      f"centroid={metrics['spectral_centroid_hz']:.0f}Hz  "
                      f"flatness={metrics['spectral_flatness']:.3f}  "
                      f"tv={metrics['temporal_variance']:.3f}", flush=True)

                # 4g. Spectrogram PNG
                spec_path = output_dir / safe_id
                _save_spectrogram(wav, sr, spec_path)
                record["spectrogram_path"] = str(spec_path.with_suffix(".png"))

                record["status"] = "completed"

            except Exception as e:
                record["status"] = "failed"
                record["error"]  = str(e)
                print(f"[LoRA Evaluator] '{adapter_id}' failed: {e}", flush=True)
                traceback.print_exc()

            finally:
                # Free generator copy immediately — large model, many adapters
                try:
                    del generator
                except NameError:
                    pass
                soft_empty_cache()

            summary["adapters"].append(record)
            _write_summary()
            pbar.update(1)

        # ------------------------------------------------------------------
        # 5. Finalise summary
        # ------------------------------------------------------------------
        summary["completed_at"] = datetime.now(timezone.utc).isoformat()
        _write_summary()
        print(f"\n[LoRA Evaluator] Done. Summary: {summary_path}", flush=True)

        # ------------------------------------------------------------------
        # 6. Comparison chart
        # ------------------------------------------------------------------
        completed = [r for r in summary["adapters"] if r.get("status") == "completed"]
        if completed:
            ids          = [r["id"] for r in completed]
            metrics_list = [r["spectral_metrics"] for r in completed]
            chart_path   = output_dir / "metric_comparison.png"
            comparison   = _draw_metric_comparison(ids, metrics_list, chart_path)
            print(f"[LoRA Evaluator] Comparison chart: {chart_path}", flush=True)
        else:
            from PIL import Image
            comparison = _pil_to_tensor(Image.new("RGB", (400, 200), (255, 255, 255)))

        return (str(summary_path), comparison)
