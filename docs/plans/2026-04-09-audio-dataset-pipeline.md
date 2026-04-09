# Audio Dataset Pipeline Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add 5 chainable ComfyUI nodes for in-memory audio dataset preprocessing: load → resample → LUFS normalize → inspect/filter → extract single item.

**Architecture:** Single new file `nodes/selva_dataset_pipeline.py` defines a custom `AUDIO_DATASET` type (list of dicts) and all 5 node classes. Nodes are stateless transforms — each takes `AUDIO_DATASET` and returns `AUDIO_DATASET`. No disk I/O except in the Loader. Register all nodes in `nodes/__init__.py`.

**Tech Stack:** `pyloudnorm` (BS.1770-4 LUFS), `soxr` (VHQ resampling), `torchaudio`, `torch`. Both confirmed present in the ComfyUI environment at `/media/p5/miniforge3/envs/latestcomfyui`.

---

## The `AUDIO_DATASET` type

Used as the ComfyUI type string `"AUDIO_DATASET"`. At runtime it is a Python list of dicts:

```python
[
    {
        "waveform": torch.Tensor,  # shape [1, C, L], float32, range [-1, 1]
        "sample_rate": int,
        "name": str,               # original filename stem, for reporting
    },
    ...
]
```

---

### Task 1: Create the file skeleton and AUDIO_DATASET constant

**Files:**
- Create: `nodes/selva_dataset_pipeline.py`

**Step 1: Write the file with imports and type constant only**

```python
"""SelVA Audio Dataset Pipeline — chainable in-memory preprocessing nodes.

Typical chain:
  SelvaDatasetLoader
      ↓ AUDIO_DATASET
  SelvaDatasetResampler       (optional)
      ↓ AUDIO_DATASET
  SelvaDatasetLUFSNormalizer  (optional)
      ↓ AUDIO_DATASET
  SelvaDatasetInspector       (optional)
      ↓ AUDIO_DATASET  +  STRING report
  SelvaDatasetItemExtractor   → AUDIO (bridges to save/preview nodes)
"""

from pathlib import Path

import numpy as np
import torch
import torchaudio

from .utils import SELVA_CATEGORY

# ComfyUI custom type name — passed between all dataset pipeline nodes
AUDIO_DATASET = "AUDIO_DATASET"

_AUDIO_EXTS = {".wav", ".flac", ".mp3", ".ogg", ".aac", ".m4a"}
```

**Step 2: Verify import works (no test framework needed — just a quick smoke check)**

```bash
cd /media/p5/Comfyui-Prismaudio
python3 -c "from nodes.selva_dataset_pipeline import AUDIO_DATASET; print(AUDIO_DATASET)"
```
Expected output: `AUDIO_DATASET`

**Step 3: Commit**

```bash
git add nodes/selva_dataset_pipeline.py
git commit -m "feat: add audio dataset pipeline skeleton"
```

---

### Task 2: SelvaDatasetLoader

**Files:**
- Modify: `nodes/selva_dataset_pipeline.py`

**Step 1: Add the Loader class**

```python
class SelvaDatasetLoader:
    """Load all audio files in a folder into an in-memory AUDIO_DATASET."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "folder": ("STRING", {
                    "default": "",
                    "tooltip": "Absolute path to folder containing audio files. Searched recursively.",
                }),
            }
        }

    RETURN_TYPES  = (AUDIO_DATASET,)
    RETURN_NAMES  = ("dataset",)
    FUNCTION      = "load"
    CATEGORY      = SELVA_CATEGORY
    DESCRIPTION   = "Load all audio files from a folder into memory as an AUDIO_DATASET."

    def load(self, folder: str):
        folder = Path(folder.strip())
        if not folder.exists():
            raise FileNotFoundError(f"[DatasetLoader] Folder not found: {folder}")

        files = [f for f in folder.rglob("*") if f.suffix.lower() in _AUDIO_EXTS]
        if not files:
            raise RuntimeError(f"[DatasetLoader] No audio files found in {folder}")

        dataset = []
        for f in sorted(files):
            try:
                wav, sr = torchaudio.load(str(f))          # [C, L]
                wav = wav.unsqueeze(0).float()              # [1, C, L]
                dataset.append({"waveform": wav, "sample_rate": sr, "name": f.stem})
            except Exception as e:
                print(f"[DatasetLoader] Skipping {f.name}: {e}", flush=True)

        print(f"[DatasetLoader] Loaded {len(dataset)} clips from {folder}", flush=True)
        return (dataset,)
```

**Step 2: Smoke test**

```bash
python3 -c "
from nodes.selva_dataset_pipeline import SelvaDatasetLoader
node = SelvaDatasetLoader()
ds, = node.load('/media/unraid/davinci/Selva/BJ')
print(len(ds), 'clips', ds[0]['name'], ds[0]['waveform'].shape, ds[0]['sample_rate'])
"
```
Expected: prints clip count, first clip name, shape like `torch.Size([1, 2, 352800])`, sample rate.

**Step 3: Commit**

```bash
git add nodes/selva_dataset_pipeline.py
git commit -m "feat: add SelvaDatasetLoader node"
```

---

### Task 3: SelvaDatasetResampler

**Files:**
- Modify: `nodes/selva_dataset_pipeline.py`

**Step 1: Add the Resampler class**

Uses `soxr` directly for VHQ quality. `soxr.resample` operates on numpy arrays, shape `[L, C]` (time-first).

```python
class SelvaDatasetResampler:
    """Resample all clips in a dataset to a target sample rate using soxr VHQ."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "dataset": (AUDIO_DATASET,),
                "target_sr": ("INT", {
                    "default": 44100, "min": 8000, "max": 192000,
                    "tooltip": "Target sample rate. 44100 for large SelVA model, 16000 for small.",
                }),
            }
        }

    RETURN_TYPES  = (AUDIO_DATASET,)
    RETURN_NAMES  = ("dataset",)
    FUNCTION      = "resample"
    CATEGORY      = SELVA_CATEGORY
    DESCRIPTION   = "Resample all clips to target_sr using soxr VHQ. Skips clips already at target rate."

    def resample(self, dataset, target_sr: int):
        import soxr

        out = []
        changed = 0
        for item in dataset:
            sr = item["sample_rate"]
            if sr == target_sr:
                out.append(item)
                continue

            wav = item["waveform"][0]               # [C, L]
            # soxr expects [L, C] (time-first), float64
            wav_np = wav.permute(1, 0).double().numpy()   # [L, C]
            wav_rs = soxr.resample(wav_np, sr, target_sr, quality="VHQ")
            wav_t  = torch.from_numpy(wav_rs).float().permute(1, 0).unsqueeze(0)  # [1, C, L]
            out.append({"waveform": wav_t, "sample_rate": target_sr, "name": item["name"]})
            changed += 1

        print(f"[DatasetResampler] {changed}/{len(dataset)} clips resampled → {target_sr} Hz", flush=True)
        return (out,)
```

**Step 2: Smoke test**

```bash
python3 -c "
from nodes.selva_dataset_pipeline import SelvaDatasetLoader, SelvaDatasetResampler
ds, = SelvaDatasetLoader().load('/media/unraid/davinci/Selva/BJ')
ds2, = SelvaDatasetResampler().resample(ds, 44100)
print('ok', ds2[0]['sample_rate'], ds2[0]['waveform'].shape)
"
```

**Step 3: Commit**

```bash
git add nodes/selva_dataset_pipeline.py
git commit -m "feat: add SelvaDatasetResampler node (soxr VHQ)"
```

---

### Task 4: SelvaDatasetLUFSNormalizer

**Files:**
- Modify: `nodes/selva_dataset_pipeline.py`

**Step 1: Add the LUFS normalizer class**

`pyloudnorm.Meter` requires numpy float64 array shape `[L]` (mono) or `[L, C]` (multichannel, channels last). True peak limit applied after gain.

```python
class SelvaDatasetLUFSNormalizer:
    """Normalize each clip to a target integrated LUFS level + true peak limit."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "dataset": (AUDIO_DATASET,),
                "target_lufs": ("FLOAT", {
                    "default": -23.0, "min": -40.0, "max": -6.0, "step": 0.5,
                    "tooltip": "Target integrated loudness in LUFS. -23 is EBU R128 standard.",
                }),
                "true_peak_dbtp": ("FLOAT", {
                    "default": -1.0, "min": -6.0, "max": 0.0, "step": 0.5,
                    "tooltip": "True peak ceiling in dBTP. Applied after LUFS gain.",
                }),
            }
        }

    RETURN_TYPES  = (AUDIO_DATASET,)
    RETURN_NAMES  = ("dataset",)
    FUNCTION      = "normalize"
    CATEGORY      = SELVA_CATEGORY
    DESCRIPTION   = (
        "Normalize each clip to target_lufs (BS.1770-4) then apply a true peak ceiling. "
        "Skips clips that are too short for LUFS measurement (< 0.4 s)."
    )

    def normalize(self, dataset, target_lufs: float, true_peak_dbtp: float):
        import pyloudnorm as pyln

        tp_linear = 10.0 ** (true_peak_dbtp / 20.0)
        out = []
        skipped = 0

        for item in dataset:
            wav = item["waveform"][0]           # [C, L]
            sr  = item["sample_rate"]

            # pyloudnorm wants [L] mono or [L, C] multichannel, float64
            wav_np = wav.permute(1, 0).double().numpy()  # [L, C]
            if wav_np.shape[1] == 1:
                wav_np = wav_np[:, 0]           # [L] mono

            meter = pyln.Meter(sr)
            try:
                loudness = meter.integrated_loudness(wav_np)
            except Exception:
                skipped += 1
                out.append(item)
                continue

            if not np.isfinite(loudness):
                skipped += 1
                out.append(item)
                continue

            gain_db     = target_lufs - loudness
            gain_linear = 10.0 ** (gain_db / 20.0)

            wav_norm = wav * gain_linear

            # True peak limit
            peak = wav_norm.abs().max().item()
            if peak > tp_linear:
                wav_norm = wav_norm * (tp_linear / peak)

            out.append({"waveform": wav_norm.unsqueeze(0), "sample_rate": sr, "name": item["name"]})

        print(
            f"[LUFSNormalizer] {len(dataset) - skipped}/{len(dataset)} clips normalized  "
            f"target={target_lufs} LUFS  TP={true_peak_dbtp} dBTP  skipped={skipped}",
            flush=True,
        )
        return (out,)
```

**Step 2: Smoke test**

```bash
python3 -c "
from nodes.selva_dataset_pipeline import SelvaDatasetLoader, SelvaDatasetLUFSNormalizer
ds, = SelvaDatasetLoader().load('/media/unraid/davinci/Selva/BJ')
ds2, = SelvaDatasetLUFSNormalizer().normalize(ds, -23.0, -1.0)
print('ok', ds2[0]['name'], ds2[0]['waveform'].abs().max().item())
"
```
Expected: peak ≤ ~0.89 (≈ -1 dBTP).

**Step 3: Commit**

```bash
git add nodes/selva_dataset_pipeline.py
git commit -m "feat: add SelvaDatasetLUFSNormalizer node (pyloudnorm BS.1770-4)"
```

---

### Task 5: SelvaDatasetInspector

**Files:**
- Modify: `nodes/selva_dataset_pipeline.py`

**Step 1: Add helper functions for artifact detection**

```python
def _check_hf_shelf(wav: torch.Tensor, sr: int) -> bool:
    """Return True if clip looks codec-compressed (hard HF shelf above 15 kHz).

    Method: compare mean energy in 1–5 kHz band vs 15–20 kHz band via STFT.
    A ratio > 40 dB (i.e. near-silence above 15 kHz) flags codec artifacts.
    """
    if sr < 32000:
        return False  # can't assess HF at low sample rates

    n_fft  = 2048
    hop    = 512
    window = torch.hann_window(n_fft)
    mono   = wav[0].mean(0)  # [L]
    stft   = torch.stft(mono, n_fft, hop, n_fft, window, return_complex=True)
    mag_sq = stft.abs().pow(2).mean(-1)  # [n_freqs]

    freqs     = torch.linspace(0, sr / 2, n_fft // 2 + 1)
    band_lo   = (freqs >= 1000) & (freqs < 5000)
    band_hi   = (freqs >= 15000) & (freqs < 20000)

    if band_hi.sum() == 0:
        return False

    energy_lo = mag_sq[band_lo].mean().clamp(min=1e-12)
    energy_hi = mag_sq[band_hi].mean().clamp(min=1e-12)
    ratio_db  = 10.0 * torch.log10(energy_lo / energy_hi).item()
    return ratio_db > 40.0


def _estimate_snr(wav: torch.Tensor) -> float:
    """Rough SNR estimate: ratio of 95th-percentile frame RMS to 5th-percentile frame RMS."""
    mono   = wav[0].mean(0)  # [L]
    frames = mono.unfold(0, 2048, 512)          # [N, 2048]
    rms    = frames.pow(2).mean(-1).sqrt()      # [N]
    p95    = torch.quantile(rms, 0.95).item()
    p05    = torch.quantile(rms, 0.05).clamp(min=1e-8).item()
    return 20.0 * np.log10(p95 / p05 + 1e-8)
```

**Step 2: Add the Inspector class**

```python
class SelvaDatasetInspector:
    """Analyze each clip for quality issues and optionally filter out flagged clips."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "dataset": (AUDIO_DATASET,),
                "skip_rejected": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "If True, flagged clips are removed from the output dataset. "
                               "If False, all clips pass through but the report still lists issues.",
                }),
                "min_snr_db": ("FLOAT", {
                    "default": 15.0, "min": 0.0, "max": 60.0, "step": 1.0,
                    "tooltip": "Clips with estimated SNR below this value are flagged.",
                }),
                "check_codec_artifacts": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Flag clips with a hard HF shelf above 15 kHz (MP3/codec artifact signature).",
                }),
            }
        }

    RETURN_TYPES  = (AUDIO_DATASET, "STRING")
    RETURN_NAMES  = ("dataset", "report")
    FUNCTION      = "inspect"
    CATEGORY      = SELVA_CATEGORY
    DESCRIPTION   = (
        "Analyze each clip for clipping, low SNR, and codec artifacts. "
        "Outputs a filtered AUDIO_DATASET and a text report. "
        "Connect report to a ShowText node to preview in the UI."
    )

    def inspect(self, dataset, skip_rejected: bool, min_snr_db: float, check_codec_artifacts: bool):
        clean   = []
        flagged = []
        lines   = ["SelVA Dataset Inspector Report", "=" * 40]

        for item in dataset:
            wav    = item["waveform"]
            sr     = item["sample_rate"]
            name   = item["name"]
            issues = []

            # Clipping
            peak = wav.abs().max().item()
            if peak > 0.99:
                issues.append(f"clipping (peak={peak:.3f})")

            # Low SNR
            snr = _estimate_snr(wav)
            if snr < min_snr_db:
                issues.append(f"low SNR ({snr:.1f} dB < {min_snr_db} dB)")

            # Codec artifacts
            if check_codec_artifacts and _check_hf_shelf(wav, sr):
                issues.append("codec artifact (HF shelf > 15 kHz)")

            if issues:
                flagged.append(name)
                lines.append(f"  FLAGGED  {name}: {', '.join(issues)}")
                if not skip_rejected:
                    clean.append(item)
            else:
                clean.append(item)
                lines.append(f"  OK       {name}")

        lines.append("=" * 40)
        lines.append(
            f"Total: {len(dataset)}  Clean: {len(clean)}  Flagged: {len(flagged)}"
            + (" (removed)" if skip_rejected else " (kept)")
        )
        report = "\n".join(lines)
        print(f"[DatasetInspector]\n{report}", flush=True)
        return (clean, report)
```

**Step 3: Smoke test**

```bash
python3 -c "
from nodes.selva_dataset_pipeline import SelvaDatasetLoader, SelvaDatasetInspector
ds, = SelvaDatasetLoader().load('/media/unraid/davinci/Selva/BJ')
clean, report = SelvaDatasetInspector().inspect(ds, skip_rejected=False, min_snr_db=15.0, check_codec_artifacts=True)
print(report)
"
```
Expected: report with per-clip OK/FLAGGED lines and summary counts.

**Step 4: Commit**

```bash
git add nodes/selva_dataset_pipeline.py
git commit -m "feat: add SelvaDatasetInspector node (codec artifacts, SNR, clipping)"
```

---

### Task 6: SelvaDatasetItemExtractor

**Files:**
- Modify: `nodes/selva_dataset_pipeline.py`

**Step 1: Add the extractor class**

```python
class SelvaDatasetItemExtractor:
    """Extract a single AUDIO item from an AUDIO_DATASET by index.

    Bridges the dataset pipeline to any node that accepts a standard AUDIO
    input — save audio, HF Smoother, Spectral Matcher, etc.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "dataset": (AUDIO_DATASET,),
                "index":   ("INT", {
                    "default": 0, "min": 0, "max": 9999,
                    "tooltip": "0-based index. Wraps around if index >= dataset length.",
                }),
            }
        }

    RETURN_TYPES  = ("AUDIO", "STRING", "INT")
    RETURN_NAMES  = ("audio",  "name",   "total")
    FUNCTION      = "extract"
    CATEGORY      = SELVA_CATEGORY
    DESCRIPTION   = (
        "Extract one clip from an AUDIO_DATASET by index. "
        "Returns standard AUDIO (compatible with all audio nodes), "
        "the clip name, and the total dataset length."
    )

    def extract(self, dataset, index: int):
        if not dataset:
            raise RuntimeError("[DatasetItemExtractor] Dataset is empty.")
        idx  = index % len(dataset)
        item = dataset[idx]
        audio = {"waveform": item["waveform"], "sample_rate": item["sample_rate"]}
        print(
            f"[DatasetItemExtractor] [{idx}/{len(dataset)-1}] {item['name']}  "
            f"sr={item['sample_rate']}  shape={tuple(item['waveform'].shape)}",
            flush=True,
        )
        return (audio, item["name"], len(dataset))
```

**Step 2: Smoke test**

```bash
python3 -c "
from nodes.selva_dataset_pipeline import SelvaDatasetLoader, SelvaDatasetItemExtractor
ds, = SelvaDatasetLoader().load('/media/unraid/davinci/Selva/BJ')
audio, name, total = SelvaDatasetItemExtractor().extract(ds, 0)
print(name, total, audio['waveform'].shape, audio['sample_rate'])
"
```

**Step 3: Commit**

```bash
git add nodes/selva_dataset_pipeline.py
git commit -m "feat: add SelvaDatasetItemExtractor node"
```

---

### Task 7: Register all nodes in __init__.py

**Files:**
- Modify: `nodes/__init__.py:4-25`

**Step 1: Add the 5 new entries to `_NODES`**

Add inside the `_NODES` dict, after `"SelvaDittoOptimizer"`:

```python
    "SelvaDatasetLoader":          (".selva_dataset_pipeline", "SelvaDatasetLoader",          "SelVA Dataset Loader"),
    "SelvaDatasetResampler":       (".selva_dataset_pipeline", "SelvaDatasetResampler",       "SelVA Dataset Resampler"),
    "SelvaDatasetLUFSNormalizer":  (".selva_dataset_pipeline", "SelvaDatasetLUFSNormalizer",  "SelVA Dataset LUFS Normalizer"),
    "SelvaDatasetInspector":       (".selva_dataset_pipeline", "SelvaDatasetInspector",       "SelVA Dataset Inspector"),
    "SelvaDatasetItemExtractor":   (".selva_dataset_pipeline", "SelvaDatasetItemExtractor",   "SelVA Dataset Item Extractor"),
```

**Step 2: Verify registration**

```bash
python3 -c "
import sys; sys.path.insert(0, '/media/p5/Comfyui-Prismaudio')
from nodes import NODE_CLASS_MAPPINGS
keys = [k for k in NODE_CLASS_MAPPINGS if 'Dataset' in k]
print(keys)
"
```
Expected: list of 5 dataset node keys.

**Step 3: Final commit**

```bash
git add nodes/__init__.py
git commit -m "feat: register audio dataset pipeline nodes in __init__.py"
```

---

## Summary

5 nodes in `nodes/selva_dataset_pipeline.py`, all registered in `__init__.py`:

| Node | In | Out |
|------|----|-----|
| SelvaDatasetLoader | folder path | AUDIO_DATASET |
| SelvaDatasetResampler | AUDIO_DATASET | AUDIO_DATASET |
| SelvaDatasetLUFSNormalizer | AUDIO_DATASET | AUDIO_DATASET |
| SelvaDatasetInspector | AUDIO_DATASET | AUDIO_DATASET + STRING |
| SelvaDatasetItemExtractor | AUDIO_DATASET + index | AUDIO + name + total |

Dependencies: `pyloudnorm`, `soxr` — both confirmed present in the ComfyUI env.
