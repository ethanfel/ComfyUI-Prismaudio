# ComfyUI-PrismAudio Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build ComfyUI custom nodes for PrismAudio video-to-audio and text-to-audio generation with adaptive VRAM management and isolated feature extraction.

**Architecture:** Selective code extraction from PrismAudio `prismaudio` branch into `prismaudio_core/` module. 5 ComfyUI nodes (ModelLoader, FeatureLoader, FeatureExtractor, Sampler, TextOnly). Feature extraction via subprocess bridge to isolated JAX/TF environment. Auto-download from HuggingFace with gated model support.

**Tech Stack:** PyTorch, ComfyUI APIs (folder_paths, comfy.model_management, comfy.utils), HuggingFace Hub, transformers (T5-Gemma), einops, k-diffusion, safetensors

---

## Bug Fixes Applied (from review)

This plan incorporates fixes for all 14 bugs identified during review:

1. **sample_discrete_euler callback**: Copy function into prismaudio_core, add callback param to the sampling loop
2. **Metadata format**: Return `(dict,)` tuple, not flat dict — matches `MultiConditioner.forward(batch_metadata: List[Dict])`
3. **video_exist**: Use `torch.tensor(True/False)`, not Python bool
4. **None features**: Use zero tensors of correct shape, never None — `pad_sequence(None)` crashes
5. **update_seq_lengths removed**: Does not exist in source. Model adapts to input shapes dynamically — no seq length config needed
6. **Sequence length config**: Not needed — model handles variable lengths natively via input tensor shapes
7. **T5-Gemma class**: Use `AutoModelForSeq2SeqLM.get_encoder()`, not `AutoModel.encoder`
8. **Peak normalization**: Add `audio / audio.abs().max().clamp(min=1e-8)` before clamp
9. **Empty feature substitution**: Match reference approach — substitute on raw conditioning output with correct shapes
10. **hf_token security**: Remove STRING widget entirely. Rely on env var / `huggingface-cli login` only. Document in README
11. **Synchformer size**: Corrected to ~950MB in docs
12. **T5 truncation**: Match reference — `truncation=False`, no max_length
13. **Remove global_video/text_features from metadata**: Not consumed by any conditioner
14. **Add tqdm to requirements**

### Bug Fixes Applied (from second review)

15. **Sync_MLP zero-tensor crash**: Sync zero-tensor fallback must be `[8, 768]` not `[1, 768]` — Sync_MLP does `length // 8` which gives 0 for length=1, causing `F.interpolate` on empty tensor
16. **sample_discrete_euler undefined `i`**: Loop needs `enumerate()` — `for i, (t_curr, t_prev) in enumerate(zip(...))`
17. **_update_seq_lengths removed entirely**: Was a no-op (attributes don't exist on DiT). Model handles variable lengths natively — function deleted
18. **cot_description removed from Sampler**: Was dead code — features already contain pre-computed text_features
19. **Conditioner VRAM leak**: Add `diffusion.conditioner.to(get_offload_device())` after generation in offload path
20. **VAE size corrected**: ~2.52GB, not ~300MB

### Bug Fixes Applied (from third review)

21. **Remove video_features substitution**: `_substitute_empty_features` should only substitute sync_features. Reference code checks for `metaclip_features` (wrong key for prismaudio config), so video substitution never runs. Cond_MLP with zero input + bias-free linears naturally produces near-zero output
22. **Remove dead `sample()` and `sample_rf`**: Wrong noise schedule (linear vs cosine), never called for rectified_flow. Only keep `sample_discrete_euler`
23. **VAE decode in fp32**: Keep pretransform in fp32 even when rest of model is fp16/bf16 — snake activations overflow in fp16
24. **Lazy imports in nodes/__init__.py**: Use try/except to allow incremental development
25. **MPS Generator guard**: `torch.Generator(device="cpu")` on Apple Silicon, move noise to device after
26. **Use comfy.utils.load_torch_file for VAE**: Consistent with diffusion loading, handles PyTorch 2.6+ weights_only default
27. **Task 10 stale reference**: Remove mention of `_update_seq_lengths`

### Bug Fixes Applied (from fourth review)

28. **TextOnly missing MPS guard**: Fix-on-fix regression — MPS Generator guard was applied to Sampler but not TextOnly
29. **TextOnly noise dtype**: Was passing dtype to torch.randn directly (fp16 noise), now generates fp32 then converts (matching Sampler)
30. **Sync substitution seq length**: Low-severity divergence from reference, accepted (DiT handles variable-length sync_cond)

---

### Task 1: Project Scaffolding

**Files:**
- Create: `__init__.py`
- Create: `nodes/__init__.py`
- Create: `nodes/utils.py`
- Create: `requirements.txt`

**Step 1: Create requirements.txt**

```
einops>=0.7.0
safetensors
huggingface_hub
transformers>=4.52.3
k-diffusion>=0.1.1
alias-free-torch
descript-audio-codec
tqdm
```

**Step 2: Create nodes/utils.py with shared helpers**

```python
import os
import torch
import folder_paths
import comfy.model_management as mm

PRISMAUDIO_CATEGORY = "PrismAudio"
SAMPLE_RATE = 44100
DOWNSAMPLING_RATIO = 2048
IO_CHANNELS = 64

def get_prismaudio_model_dir():
    """Get or create the prismaudio model directory."""
    model_dir = os.path.join(folder_paths.models_dir, "prismaudio")
    os.makedirs(model_dir, exist_ok=True)
    return model_dir

def register_model_folder():
    """Register prismaudio model folder with ComfyUI."""
    model_dir = get_prismaudio_model_dir()
    folder_paths.add_model_folder_path("prismaudio", model_dir)

def get_device():
    return mm.get_torch_device()

def get_offload_device():
    return mm.unet_offload_device()

def get_free_memory(device=None):
    if device is None:
        device = get_device()
    return mm.get_free_memory(device)

def soft_empty_cache():
    mm.soft_empty_cache()

def determine_precision(preference, device):
    """Determine the best precision for the given device."""
    if preference != "auto":
        return {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}[preference]
    if device.type == "cpu":
        return torch.float32
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16

def determine_offload_strategy(preference):
    """Determine offload strategy based on available VRAM."""
    if preference != "auto":
        return preference
    free_mem = get_free_memory()
    gb = free_mem / (1024 ** 3)
    if gb >= 24:
        return "keep_in_vram"
    else:
        return "offload_to_cpu"

def try_import_flash_attn():
    """Try to import flash attention, return None if unavailable."""
    try:
        import flash_attn
        return flash_attn
    except ImportError:
        return None

def resolve_hf_token():
    """Resolve HF token from env var or cached login. No widget — security risk."""
    env_token = os.environ.get("HF_TOKEN")
    if env_token:
        return env_token
    # huggingface_hub will use cached token automatically if None is passed
    return None
```

**Step 3: Create nodes/__init__.py**

```python
NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

# Lazy imports — allows incremental development (nodes can be added one at a time)
_NODES = {
    "PrismAudioModelLoader": (".model_loader", "PrismAudioModelLoader", "PrismAudio Model Loader"),
    "PrismAudioFeatureLoader": (".feature_loader", "PrismAudioFeatureLoader", "PrismAudio Feature Loader"),
    "PrismAudioFeatureExtractor": (".feature_extractor", "PrismAudioFeatureExtractor", "PrismAudio Feature Extractor"),
    "PrismAudioSampler": (".sampler", "PrismAudioSampler", "PrismAudio Sampler"),
    "PrismAudioTextOnly": (".text_only", "PrismAudioTextOnly", "PrismAudio Text Only"),
}

for key, (module_path, class_name, display_name) in _NODES.items():
    try:
        import importlib
        mod = importlib.import_module(module_path, package=__name__)
        NODE_CLASS_MAPPINGS[key] = getattr(mod, class_name)
        NODE_DISPLAY_NAME_MAPPINGS[key] = display_name
    except (ImportError, AttributeError) as e:
        print(f"[PrismAudio] Skipping {key}: {e}")
```

**Step 4: Create top-level __init__.py**

```python
"""
ComfyUI-PrismAudio: Video-to-Audio and Text-to-Audio generation using PrismAudio (ICLR 2026).
"""
from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
```

**Step 5: Commit**

```bash
git add __init__.py nodes/__init__.py nodes/utils.py requirements.txt
git commit -m "feat: project scaffolding with shared utils and node registration"
```

---

### Task 2: Extract prismaudio_core — Model Config + Factory

**Files:**
- Create: `prismaudio_core/__init__.py`
- Create: `prismaudio_core/configs/prismaudio.json` (copy from PrismAudio repo)
- Create: `prismaudio_core/factory.py` (adapted from `PrismAudio/models/factory.py`)

**Step 1: Create prismaudio_core/__init__.py**

```python
"""
PrismAudio core inference modules.
Extracted from https://github.com/FunAudioLLM/ThinkSound (prismaudio branch).
Only inference-critical code — no training, no JAX/TF dependencies.
"""
```

**Step 2: Copy prismaudio.json config**

Fetch from `https://raw.githubusercontent.com/FunAudioLLM/ThinkSound/prismaudio/PrismAudio/configs/model_configs/prismaudio.json` and save to `prismaudio_core/configs/prismaudio.json`. This is a JSON config file with no code — copy verbatim.

**Step 3: Create factory.py**

Extract from `PrismAudio/models/factory.py`. Keep only these functions (remove training-related code):
- `create_model_from_config(model_config)` — entry point
- `create_diffusion_cond_from_config(config)` — creates the full model
- `create_pretransform_from_config(pretransform_config, sample_rate)` — VAE
- `create_autoencoder_from_config(config)` — AudioAutoencoder
- `create_bottleneck_from_config(config)` — VAEBottleneck
- `create_multi_conditioner_from_conditioning_config(config)` — conditioners

All imports should reference `prismaudio_core.models.*` instead of `PrismAudio.models.*`.

**Step 4: Commit**

```bash
git add prismaudio_core/
git commit -m "feat: extract prismaudio_core config and model factory"
```

---

### Task 3: Extract prismaudio_core — Model Modules

**Files:**
- Create: `prismaudio_core/models/__init__.py`
- Create: `prismaudio_core/models/dit.py` (from `PrismAudio/models/dit.py`)
- Create: `prismaudio_core/models/diffusion.py` (from `PrismAudio/models/diffusion.py`)
- Create: `prismaudio_core/models/conditioners.py` (from `PrismAudio/models/conditioners.py`)
- Create: `prismaudio_core/models/autoencoders.py` (from `PrismAudio/models/autoencoders.py`)
- Create: `prismaudio_core/models/pretransforms.py` (from `PrismAudio/models/pretransforms.py`)
- Create: `prismaudio_core/models/blocks.py` (from `PrismAudio/models/blocks.py`)
- Create: `prismaudio_core/models/utils.py` (from `PrismAudio/models/utils.py`)
- Create: `prismaudio_core/models/bottleneck.py` (from `PrismAudio/models/bottleneck.py`)
- Create: `prismaudio_core/models/transformer.py` (from `PrismAudio/models/transformer.py`)
- Create: `prismaudio_core/models/local_attention.py` (if used by transformer)

**Step 1: Extract model files**

For each file, extract from the PrismAudio repo. Key modifications:
- Change all internal imports from `PrismAudio.models.*` to `prismaudio_core.models.*`
- Remove training-only code (loss functions, training step methods, gradient checkpointing setup)
- Keep all inference paths intact

**Critical classes to preserve:**

From `dit.py`:
- `DiffusionTransformer` — full class with `forward()`, CFG logic, conditioning assembly
- `FourierFeatures` — timestep embedding
- Keep `empty_clip_feat` and `empty_sync_feat` learned parameters (nn.Parameter, zeros)

From `diffusion.py`:
- `ConditionedDiffusionModelWrapper` — with `get_conditioning_inputs()` and routing logic
- `DiTWrapper` — thin wrapper that passes all kwargs through
- `create_diffusion_cond_from_config()` — factory function

From `conditioners.py`:
- `Cond_MLP` (type `"cond_mlp"`) — for video_features and text_features. Uses `pad_sequence`, 2-layer MLP, returns `[embeddings, ones_mask]`. During eval with batch<16, doubles batch with null embed for CFG
- `Sync_MLP` (type `"sync_mlp"`) — for sync_features with learnable `sync_pos_emb` of shape (1,1,8,dim), reshapes into segments of 8, interpolates to target length
- `MultiConditioner` — iterates over `batch_metadata: List[Dict]`, collects per-sample inputs, calls each conditioner. Returns dict of `{key: (tensor, mask)}`
- `create_multi_conditioner_from_conditioning_config()` — factory

From `autoencoders.py`:
- `AudioAutoencoder` — with `encode_audio()` and `decode_audio()`
- `OobleckEncoder`, `OobleckDecoder` — with ResidualUnit, snake activation
- Dependencies: `alias_free_torch` (SnakeBeta), `dac.nn` (WNConv1d, WNConvTranspose1d)

From `pretransforms.py`:
- `AutoencoderPretransform` — wraps AudioAutoencoder, `encode()` and `decode()` methods

From `bottleneck.py`:
- `VAEBottleneck` — reparameterization trick (split mean/logvar, sample)

From `blocks.py`:
- Any shared blocks used by the above (attention blocks, FeedForward, etc.)

From `transformer.py`:
- `ContinuousTransformer` — the core transformer with cross-attention, used by DiffusionTransformer

From `utils.py`:
- `load_ckpt_state_dict()` — handles .safetensors and .ckpt, optional prefix stripping
- `remove_weight_norm_from_model()` — used in some inference paths

**Step 2: Handle flash-attn gracefully in transformer.py / blocks.py**

Replace hard `import flash_attn` with:
```python
try:
    from flash_attn import flash_attn_func
    HAS_FLASH_ATTN = True
except ImportError:
    HAS_FLASH_ATTN = False
```

In the attention forward pass, use:
```python
if HAS_FLASH_ATTN:
    out = flash_attn_func(q, k, v, ...)
else:
    out = torch.nn.functional.scaled_dot_product_attention(q, k, v, ...)
```

**Step 3: Verify imports resolve**

Run: `python -c "from prismaudio_core.factory import create_model_from_config; print('OK')"` from the project root (with ComfyUI's python).

Expected: `OK` (or import errors to fix iteratively)

**Step 4: Commit**

```bash
git add prismaudio_core/models/
git commit -m "feat: extract prismaudio_core model modules (DiT, conditioners, VAE, diffusion)"
```

---

### Task 4: Extract prismaudio_core — Inference/Sampling (with callback fix)

**Files:**
- Create: `prismaudio_core/inference/__init__.py`
- Create: `prismaudio_core/inference/sampling.py` (MODIFIED from `PrismAudio/inference/sampling.py`)
- Create: `prismaudio_core/inference/utils.py` (from `PrismAudio/inference/utils.py`)

**Step 1: Extract sampling.py WITH callback support added**

The original `sample_discrete_euler` uses `tqdm` and has no callback parameter.
We MUST copy and modify it to add callback support for ComfyUI progress bars.

```python
import torch
from tqdm import trange

def sample_discrete_euler(model, x, steps, sigma_max=1, callback=None, **extra_args):
    """Discrete Euler sampler for rectified flow, with optional callback.

    Modified from PrismAudio to add callback parameter for ComfyUI progress reporting.
    Original uses tqdm internally.

    Args:
        model: The diffusion model (DiTWrapper)
        x: Initial noise tensor [B, C, T]
        steps: Number of sampling steps
        sigma_max: Maximum sigma (default 1.0 for rectified flow)
        callback: Optional callable({"i": step, "x": current_x}) for progress
        **extra_args: Passed to model() — includes cross_attn_cond, add_cond,
                      sync_cond, cfg_scale, batch_cfg, etc.
    """
    t = torch.linspace(sigma_max, 0, steps + 1, device=x.device, dtype=x.dtype)

    for i, (t_curr, t_next) in enumerate(zip(t[:-1], t[1:])):
        dt = t_next - t_curr
        t_curr_tensor = t_curr * torch.ones(x.shape[0], dtype=x.dtype, device=x.device)
        x = x + dt * model(x, t_curr_tensor, **extra_args)
        if callback is not None:
            callback({"i": i, "x": x})

    return x



# Note: sample_rf() and sample() (v-diffusion) are NOT included.
# PrismAudio uses rectified_flow objective which only needs sample_discrete_euler.
# Including unused samplers with potentially wrong math is a maintenance hazard.
```

**Step 2: Extract inference/utils.py**

Keep:
- `set_audio_channels(audio, target_channels)`
- `prepare_audio(audio, in_sr, target_sr, target_length, target_channels, device)`

**Step 3: Verify sampling import**

Run: `python -c "from prismaudio_core.inference.sampling import sample_discrete_euler; print('OK')"`

Expected: `OK`

**Step 4: Commit**

```bash
git add prismaudio_core/inference/
git commit -m "feat: extract prismaudio_core inference with callback-enabled sampling"
```

---

### Task 5: PrismAudioModelLoader Node

**Files:**
- Create: `nodes/model_loader.py`

**Step 1: Write the node**

Key design decisions:
- No hf_token widget (security risk — saved to workflow JSON). Uses env var / cached login only.
- Creates model with default config. Duration-dependent seq lengths handled at sample time.
- The model config's `sample_size: 397312` corresponds to ~9s default. For other durations,
  the Sampler node will update seq lengths on the DiT before each generation.

```python
import os
import json
import torch
import folder_paths
import comfy.model_management as mm
import comfy.utils

from .utils import (
    PRISMAUDIO_CATEGORY, get_prismaudio_model_dir, register_model_folder,
    get_device, get_offload_device, determine_precision, determine_offload_strategy,
    soft_empty_cache, resolve_hf_token,
)

# HuggingFace repo for auto-download
HF_REPO_ID = "FunAudioLLM/PrismAudio"
REQUIRED_FILES = {
    "diffusion": "prismaudio.ckpt",
    "vae": "vae.ckpt",
    "synchformer": "synchformer_state_dict.pth",
}


def _download_if_missing(filename, model_dir, hf_token=None):
    """Download a model file from HuggingFace if not present locally."""
    filepath = os.path.join(model_dir, filename)
    if os.path.exists(filepath):
        return filepath

    from huggingface_hub import hf_hub_download
    print(f"[PrismAudio] Downloading {filename} from {HF_REPO_ID}...")
    try:
        downloaded = hf_hub_download(
            repo_id=HF_REPO_ID,
            filename=filename,
            local_dir=model_dir,
            token=hf_token or None,
        )
        return downloaded
    except Exception as e:
        if "401" in str(e) or "403" in str(e) or "gated" in str(e).lower():
            raise RuntimeError(
                f"[PrismAudio] Model '{filename}' requires license acceptance. "
                f"Visit https://huggingface.co/{HF_REPO_ID} to accept the license, "
                f"then set HF_TOKEN env var or run: huggingface-cli login"
            ) from e
        raise


class PrismAudioModelLoader:
    @classmethod
    def INPUT_TYPES(cls):
        register_model_folder()
        return {
            "required": {
                "precision": (["auto", "fp32", "fp16", "bf16"],),
                "offload_strategy": (["auto", "keep_in_vram", "offload_to_cpu"],),
            },
        }

    RETURN_TYPES = ("PRISMAUDIO_MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "load_model"
    CATEGORY = PRISMAUDIO_CATEGORY

    def load_model(self, precision, offload_strategy):
        device = get_device()
        dtype = determine_precision(precision, device)
        strategy = determine_offload_strategy(offload_strategy)
        token = resolve_hf_token()
        model_dir = get_prismaudio_model_dir()

        # Auto-download missing files
        for key, filename in REQUIRED_FILES.items():
            _download_if_missing(filename, model_dir, hf_token=token)

        # Load config
        config_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "prismaudio_core", "configs", "prismaudio.json"
        )
        with open(config_path) as f:
            model_config = json.load(f)

        # Create model from config
        from prismaudio_core.factory import create_model_from_config
        model = create_model_from_config(model_config)

        # Load diffusion weights
        diffusion_path = os.path.join(model_dir, REQUIRED_FILES["diffusion"])
        diffusion_state = comfy.utils.load_torch_file(diffusion_path)
        # Handle wrapped state dicts: some ckpts wrap in {"state_dict": ...}
        if "state_dict" in diffusion_state:
            diffusion_state = diffusion_state["state_dict"]
        model.load_state_dict(diffusion_state, strict=False)

        # Load VAE weights separately
        # Use comfy.utils.load_torch_file for consistency and PyTorch 2.6+ compat
        vae_path = os.path.join(model_dir, REQUIRED_FILES["vae"])
        vae_full_state = comfy.utils.load_torch_file(vae_path)
        # Strip "autoencoder." prefix from keys
        vae_state = {}
        prefix = "autoencoder."
        for k, v in vae_full_state.items():
            if k.startswith(prefix):
                vae_state[k[len(prefix):]] = v
            else:
                vae_state[k] = v
        model.pretransform.load_state_dict(vae_state)

        # Apply precision: DiT + conditioners in user-selected dtype,
        # but keep VAE (pretransform) in fp32 to avoid NaN from snake activations in fp16
        model.model.to(dtype)        # DiTWrapper
        model.conditioner.to(dtype)  # MultiConditioner
        # model.pretransform stays in fp32

        if strategy == "keep_in_vram":
            model = model.to(device)
        else:
            model = model.to(get_offload_device())

        model.eval()

        return ({
            "model": model,
            "dtype": dtype,
            "strategy": strategy,
            "config": model_config,
            "model_dir": model_dir,
        },)
```

**Step 2: Test that ComfyUI discovers the node**

Run ComfyUI and check that "PrismAudio Model Loader" appears in the node list.

**Step 3: Commit**

```bash
git add nodes/model_loader.py
git commit -m "feat: PrismAudioModelLoader node with auto-download and adaptive VRAM"
```

---

### Task 6: PrismAudioFeatureLoader Node

**Files:**
- Create: `nodes/feature_loader.py`

**Step 1: Write the node**

```python
import os
import numpy as np
import torch
from .utils import PRISMAUDIO_CATEGORY

# Keys consumed by the conditioners (video_features, text_features, sync_features)
# global_video_features and global_text_features are NOT consumed by any conditioner
# in the prismaudio.json config — they are unused.
REQUIRED_KEYS = [
    "video_features",
    "text_features",
    "sync_features",
]


class PrismAudioFeatureLoader:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "npz_path": ("STRING", {"default": "", "tooltip": "Path to pre-computed .npz feature file"}),
            },
        }

    RETURN_TYPES = ("PRISMAUDIO_FEATURES",)
    RETURN_NAMES = ("features",)
    FUNCTION = "load_features"
    CATEGORY = PRISMAUDIO_CATEGORY

    def load_features(self, npz_path):
        if not os.path.exists(npz_path):
            raise FileNotFoundError(f"[PrismAudio] Feature file not found: {npz_path}")

        data = np.load(npz_path, allow_pickle=True)

        features = {}
        for key in REQUIRED_KEYS:
            if key in data:
                features[key] = torch.from_numpy(data[key]).float()
            else:
                print(f"[PrismAudio] Warning: key '{key}' not found in {npz_path}, using zeros")
                # Provide zero tensor rather than None — Cond_MLP/Sync_MLP crash on None
                # Sync_MLP requires length divisible by 8 (segments of 8 frames)
                if key == "sync_features":
                    features[key] = torch.zeros(8, 768)
                else:
                    features[key] = torch.zeros(1, 1024)

        # Load duration if present
        if "duration" in data:
            features["duration"] = float(data["duration"])

        return (features,)
```

**Step 2: Commit**

```bash
git add nodes/feature_loader.py
git commit -m "feat: PrismAudioFeatureLoader node for pre-computed .npz files"
```

---

### Task 7: PrismAudioFeatureExtractor Node (Subprocess Bridge)

**Files:**
- Create: `nodes/feature_extractor.py`
- Create: `scripts/extract_features.py`
- Create: `scripts/environment.yml`

**Step 1: Create the conda environment.yml**

```yaml
name: prismaudio-extract
channels:
  - conda-forge
  - defaults
dependencies:
  - python=3.10
  - pip
  - ffmpeg<7
  - pip:
    - torch>=2.6.0
    - torchaudio>=2.6.0
    - torchvision>=0.21.0
    - tensorflow-cpu==2.15.0
    - jax
    - jaxlib
    - transformers>=4.52.3
    - decord
    - einops>=0.7.0
    - numpy
    - mediapy
    - git+https://github.com/google-deepmind/videoprism.git
```

**Step 2: Create scripts/extract_features.py**

This is a standalone script that:
1. Takes `--video`, `--cot_text`, `--output` arguments
2. Loads VideoPrism, T5-Gemma, Synchformer
3. Extracts features from the video
4. Saves as `.npz`

```python
#!/usr/bin/env python3
"""
Standalone PrismAudio feature extraction script.
Run in a separate conda env with JAX/TF installed.

Usage:
    python extract_features.py --video input.mp4 --cot_text "description..." --output features.npz

Setup:
    conda env create -f environment.yml
    conda activate prismaudio-extract
"""

import argparse
import os
import sys
import numpy as np
import torch


def main():
    parser = argparse.ArgumentParser(description="PrismAudio feature extraction")
    parser.add_argument("--video", required=True, help="Path to input video")
    parser.add_argument("--cot_text", required=True, help="Chain-of-thought description")
    parser.add_argument("--output", required=True, help="Output .npz path")
    parser.add_argument("--synchformer_ckpt", default=None, help="Path to synchformer checkpoint")
    parser.add_argument("--vae_config", default=None, help="Path to VAE config JSON")
    parser.add_argument("--clip_fps", type=float, default=4.0)
    parser.add_argument("--clip_size", type=int, default=288)
    parser.add_argument("--sync_fps", type=float, default=25.0)
    parser.add_argument("--sync_size", type=int, default=224)
    args = parser.parse_args()

    if not os.path.exists(args.video):
        print(f"Error: Video not found: {args.video}")
        sys.exit(1)

    # Import feature extraction utils (requires JAX/TF)
    from data_utils.v2a_utils.feature_utils_288 import FeaturesUtils
    import torchvision.transforms as T
    from decord import VideoReader, cpu

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize feature extractor
    feat_utils = FeaturesUtils(
        vae_config_path=args.vae_config,
        synchformer_ckpt=args.synchformer_ckpt,
        device=device,
    )

    # Load and preprocess video
    vr = VideoReader(args.video, ctx=cpu(0))
    fps = vr.get_avg_fps()
    total_frames = len(vr)
    duration = total_frames / fps

    # Extract CLIP frames (4fps, 288x288)
    clip_indices = [int(i * fps / args.clip_fps) for i in range(int(duration * args.clip_fps))]
    clip_indices = [min(i, total_frames - 1) for i in clip_indices]
    clip_frames = vr.get_batch(clip_indices).asnumpy()

    clip_transform = T.Compose([
        T.ToPILImage(),
        T.Resize(args.clip_size),
        T.CenterCrop(args.clip_size),
        T.ToTensor(),
        T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    clip_input = torch.stack([clip_transform(f) for f in clip_frames]).unsqueeze(0).to(device)

    # Extract Sync frames (25fps, 224x224)
    sync_indices = [int(i * fps / args.sync_fps) for i in range(int(duration * args.sync_fps))]
    sync_indices = [min(i, total_frames - 1) for i in sync_indices]
    sync_frames = vr.get_batch(sync_indices).asnumpy()

    sync_transform = T.Compose([
        T.ToPILImage(),
        T.Resize(args.sync_size),
        T.CenterCrop(args.sync_size),
        T.ToTensor(),
        T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    sync_input = torch.stack([sync_transform(f) for f in sync_frames]).unsqueeze(0).to(device)

    # Extract features
    print("[PrismAudio] Encoding text with T5-Gemma...")
    text_features = feat_utils.encode_t5_text([args.cot_text])

    print("[PrismAudio] Encoding video with VideoPrism...")
    global_video_features, video_features, global_text_features = \
        feat_utils.encode_video_and_text_with_videoprism(clip_input, [args.cot_text])

    print("[PrismAudio] Encoding video with Synchformer...")
    sync_features = feat_utils.encode_video_with_sync(sync_input)

    # Save as .npz
    np.savez(
        args.output,
        video_features=video_features.cpu().numpy(),
        global_video_features=global_video_features.cpu().numpy(),
        text_features=text_features.cpu().numpy(),
        global_text_features=global_text_features.cpu().numpy(),
        sync_features=sync_features.cpu().numpy(),
        caption_cot=args.cot_text,
        duration=duration,
    )
    print(f"[PrismAudio] Features saved to {args.output}")


if __name__ == "__main__":
    main()
```

**Step 3: Create the feature extractor node**

```python
import os
import hashlib
import subprocess
import tempfile
import torch

from .utils import PRISMAUDIO_CATEGORY
from .feature_loader import PrismAudioFeatureLoader


def _hash_inputs(video_tensor, cot_text):
    """Create a hash of the inputs for caching."""
    h = hashlib.sha256()
    h.update(video_tensor.cpu().numpy().tobytes()[:1024 * 1024])  # First 1MB for speed
    h.update(cot_text.encode())
    return h.hexdigest()[:16]


def _save_video_tensor_to_mp4(video_tensor, output_path, fps=30):
    """Save ComfyUI IMAGE tensor [T,H,W,C] to MP4."""
    import torchvision.io as tvio
    # ComfyUI IMAGE is [T,H,W,C] float32 [0,1]
    frames = (video_tensor * 255).to(torch.uint8)
    # torchvision write_video expects [T,H,W,C] uint8
    tvio.write_video(output_path, frames, fps=fps)


class PrismAudioFeatureExtractor:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video": ("IMAGE",),
                "caption_cot": ("STRING", {"default": "", "multiline": True, "tooltip": "Chain-of-thought description"}),
            },
            "optional": {
                "python_env": ("STRING", {"default": "python", "tooltip": "Path to python binary with JAX/TF (e.g., /path/to/conda/envs/prismaudio-extract/bin/python)"}),
                "cache_dir": ("STRING", {"default": "", "tooltip": "Directory to cache extracted features. Empty = temp dir"}),
                "synchformer_ckpt": ("STRING", {"default": "", "tooltip": "Path to synchformer checkpoint (auto-resolved if empty)"}),
            },
        }

    RETURN_TYPES = ("PRISMAUDIO_FEATURES",)
    RETURN_NAMES = ("features",)
    FUNCTION = "extract_features"
    CATEGORY = PRISMAUDIO_CATEGORY

    def extract_features(self, video, caption_cot, python_env="python", cache_dir="", synchformer_ckpt=""):
        # Determine cache directory
        if not cache_dir:
            cache_dir = os.path.join(tempfile.gettempdir(), "prismaudio_features")
        os.makedirs(cache_dir, exist_ok=True)

        # Check cache
        cache_hash = _hash_inputs(video, caption_cot)
        cached_path = os.path.join(cache_dir, f"{cache_hash}.npz")
        if os.path.exists(cached_path):
            print(f"[PrismAudio] Using cached features: {cached_path}")
            loader = PrismAudioFeatureLoader()
            return loader.load_features(cached_path)

        # Save video to temp file
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
            tmp_video = tmp.name
        _save_video_tensor_to_mp4(video, tmp_video)

        # Build subprocess command
        script_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "scripts", "extract_features.py"
        )

        cmd = [
            python_env,
            script_path,
            "--video", tmp_video,
            "--cot_text", caption_cot,
            "--output", cached_path,
        ]
        if synchformer_ckpt:
            cmd.extend(["--synchformer_ckpt", synchformer_ckpt])

        print(f"[PrismAudio] Extracting features via subprocess...")
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=600,  # 10 minute timeout
            )
            if result.returncode != 0:
                raise RuntimeError(
                    f"[PrismAudio] Feature extraction failed:\n{result.stderr}"
                )
            print(result.stdout)
        finally:
            if os.path.exists(tmp_video):
                os.unlink(tmp_video)

        # Load the extracted features
        loader = PrismAudioFeatureLoader()
        return loader.load_features(cached_path)
```

**Step 4: Commit**

```bash
git add nodes/feature_extractor.py scripts/extract_features.py scripts/environment.yml
git commit -m "feat: PrismAudioFeatureExtractor node with subprocess bridge and conda env"
```

---

### Task 8: PrismAudioSampler Node

**Files:**
- Create: `nodes/sampler.py`

**Step 1: Write the sampler node**

This is the core node. Key fixes from review:
- Metadata is a TUPLE of dicts, not a flat dict
- video_exist is torch.tensor, not Python bool
- Empty features are zero tensors, not None
- Peak normalization before clamp
- Sequence lengths set on DiT config before sampling (matching predict.py approach)
- No callback kwarg forwarded to model — callback is handled by our modified sample_discrete_euler

```python
import torch
import comfy.model_management as mm
import comfy.utils

from .utils import (
    PRISMAUDIO_CATEGORY, SAMPLE_RATE, DOWNSAMPLING_RATIO, IO_CHANNELS,
    get_device, get_offload_device, soft_empty_cache,
)


class PrismAudioSampler:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("PRISMAUDIO_MODEL",),
                "features": ("PRISMAUDIO_FEATURES",),
                "duration": ("FLOAT", {"default": 10.0, "min": 1.0, "max": 30.0, "step": 0.1, "tooltip": "Audio duration in seconds"}),
                "steps": ("INT", {"default": 24, "min": 1, "max": 100, "tooltip": "Number of sampling steps"}),
                "cfg_scale": ("FLOAT", {"default": 5.0, "min": 1.0, "max": 20.0, "step": 0.1, "tooltip": "Classifier-free guidance scale"}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFF}),
            },
        }

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "generate"
    CATEGORY = PRISMAUDIO_CATEGORY

    def generate(self, model, features, duration, steps, cfg_scale, seed):
        device = get_device()
        dtype = model["dtype"]
        strategy = model["strategy"]
        diffusion = model["model"]

        # Compute latent dimensions
        latent_length = round(SAMPLE_RATE * duration / DOWNSAMPLING_RATIO)

        # Note: no seq length config needed — the model adapts to input tensor shapes
        # dynamically via its transformer architecture.

        # Determine if video features are present (not all zeros)
        has_video = features.get("video_features") is not None and features["video_features"].abs().sum() > 0

        # Build metadata as a TUPLE of dicts (one per batch sample)
        # MultiConditioner.forward(batch_metadata: List[Dict]) iterates over this
        sample_meta = {
            "video_features": features["video_features"].to(device, dtype=dtype),
            "text_features": features["text_features"].to(device, dtype=dtype),
            "sync_features": features["sync_features"].to(device, dtype=dtype),
            "video_exist": torch.tensor(has_video),
        }
        metadata = (sample_meta,)

        # Move model to device if offloaded
        if strategy == "offload_to_cpu":
            diffusion.model.to(device)
            diffusion.conditioner.to(device)
            soft_empty_cache()

        with torch.no_grad(), torch.amp.autocast(device_type=device.type, dtype=dtype):
            # Run conditioning
            conditioning = diffusion.conditioner(metadata, device)

            # Handle missing video: substitute learned empty embeddings
            if not has_video:
                _substitute_empty_features(diffusion, conditioning, device, dtype)

            # Assemble conditioning inputs for the DiT
            cond_inputs = diffusion.get_conditioning_inputs(conditioning)

            # Generate noise from seed (MPS doesn't support torch.Generator)
            gen_device = "cpu" if device.type == "mps" else device
            generator = torch.Generator(device=gen_device).manual_seed(seed)
            noise = torch.randn(
                [1, IO_CHANNELS, latent_length],
                generator=generator,
                device=gen_device,
            ).to(device=device, dtype=dtype)

            # Sample with progress bar
            pbar = comfy.utils.ProgressBar(steps)

            from prismaudio_core.inference.sampling import sample_discrete_euler

            def on_step(info):
                pbar.update(1)

            fakes = sample_discrete_euler(
                diffusion.model,
                noise,
                steps,
                callback=on_step,
                **cond_inputs,
                cfg_scale=cfg_scale,
                batch_cfg=True,
            )

            # Offload diffusion model and conditioner before VAE decode
            if strategy == "offload_to_cpu":
                diffusion.model.to(get_offload_device())
                diffusion.conditioner.to(get_offload_device())
                soft_empty_cache()
                diffusion.pretransform.to(device)

            # VAE decode in fp32 (snake activations overflow in fp16)
            with torch.amp.autocast(device_type=device.type, enabled=False):
                audio = diffusion.pretransform.decode(fakes.float())

            # Offload VAE
            if strategy == "offload_to_cpu":
                diffusion.pretransform.to(get_offload_device())
                soft_empty_cache()

        # Peak normalize then clamp (matching reference: div by max abs before clamp)
        audio = audio.float()
        peak = audio.abs().max().clamp(min=1e-8)
        audio = (audio / peak).clamp(-1, 1)

        # Return as ComfyUI AUDIO: {"waveform": [B, channels, samples], "sample_rate": int}
        return ({"waveform": audio.cpu(), "sample_rate": SAMPLE_RATE},)


def _substitute_empty_features(diffusion, conditioning, device, dtype):
    """Replace sync conditioning with learned empty embedding when video is absent.

    Only substitutes sync_features — NOT video_features. The reference code
    (predict.py/app.py) checks for 'metaclip_features' which doesn't exist in the
    prismaudio.json config, so video substitution never runs. Cond_MLP with zero
    input + bias-free linear layers naturally produces near-zero output.

    The conditioner returns {key: [tensor, mask]} where tensor is [B, seq, dim].
    """
    dit = diffusion.model.model if hasattr(diffusion.model, 'model') else diffusion.model

    # Only substitute sync_features (matching reference behavior for prismaudio config)
    if hasattr(dit, 'empty_sync_feat') and 'sync_features' in conditioning:
        empty = dit.empty_sync_feat.to(device, dtype=dtype)
        cond_tensor = conditioning['sync_features'][0]
        batch_size = cond_tensor.shape[0]
        empty_expanded = empty.unsqueeze(0).expand(batch_size, -1, -1)
        conditioning['sync_features'][0] = empty_expanded
        conditioning['sync_features'][1] = torch.ones(batch_size, 1, device=device)
```

**Step 2: Verify the node registers**

Start ComfyUI, check "PrismAudio Sampler" appears in add-node menu.

**Step 3: Commit**

```bash
git add nodes/sampler.py
git commit -m "feat: PrismAudioSampler node with correct metadata format and peak normalization"
```

---

### Task 9: PrismAudioTextOnly Node

**Files:**
- Create: `nodes/text_only.py`

**Step 1: Write the text-only node**

Key fixes from review:
- Uses `AutoModelForSeq2SeqLM.get_encoder()`, not `AutoModel.encoder`
- No truncation (matching reference)
- Metadata is tuple of dicts with torch.tensor(False) for video_exist
- Zero tensors for video/sync features, not None
- Peak normalization

```python
import torch
import comfy.model_management as mm
import comfy.utils

from .utils import (
    PRISMAUDIO_CATEGORY, SAMPLE_RATE, DOWNSAMPLING_RATIO, IO_CHANNELS,
    get_device, get_offload_device, soft_empty_cache, resolve_hf_token,
)
from .sampler import _substitute_empty_features


class PrismAudioTextOnly:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("PRISMAUDIO_MODEL",),
                "text_prompt": ("STRING", {"default": "", "multiline": True, "tooltip": "Text description for audio generation"}),
                "duration": ("FLOAT", {"default": 10.0, "min": 1.0, "max": 30.0, "step": 0.1}),
                "steps": ("INT", {"default": 24, "min": 1, "max": 100}),
                "cfg_scale": ("FLOAT", {"default": 5.0, "min": 1.0, "max": 20.0, "step": 0.1}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFF}),
            },
        }

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "generate"
    CATEGORY = PRISMAUDIO_CATEGORY

    def generate(self, model, text_prompt, duration, steps, cfg_scale, seed):
        device = get_device()
        dtype = model["dtype"]
        strategy = model["strategy"]
        diffusion = model["model"]

        latent_length = round(SAMPLE_RATE * duration / DOWNSAMPLING_RATIO)

        # Encode text with T5-Gemma
        text_features = _encode_text_t5(text_prompt, device, dtype)

        # Build metadata: tuple of one dict per sample
        # Use zero tensors for video/sync (not None — Cond_MLP crashes on None via pad_sequence)
        # Sync_MLP requires length divisible by 8 (segments of 8 frames) — minimum [8, 768]
        # These will be substituted with learned empty embeddings after conditioning
        sample_meta = {
            "video_features": torch.zeros(1, 1024, device=device, dtype=dtype),
            "text_features": text_features.to(device, dtype=dtype),
            "sync_features": torch.zeros(8, 768, device=device, dtype=dtype),
            "video_exist": torch.tensor(False),
        }
        metadata = (sample_meta,)

        if strategy == "offload_to_cpu":
            diffusion.model.to(device)
            diffusion.conditioner.to(device)
            soft_empty_cache()

        with torch.no_grad(), torch.amp.autocast(device_type=device.type, dtype=dtype):
            conditioning = diffusion.conditioner(metadata, device)

            # Substitute empty features for video/sync
            _substitute_empty_features(diffusion, conditioning, device, dtype)

            cond_inputs = diffusion.get_conditioning_inputs(conditioning)

            # Generate noise from seed (MPS doesn't support torch.Generator)
            gen_device = "cpu" if device.type == "mps" else device
            generator = torch.Generator(device=gen_device).manual_seed(seed)
            noise = torch.randn(
                [1, IO_CHANNELS, latent_length],
                generator=generator,
                device=gen_device,
            ).to(device=device, dtype=dtype)

            pbar = comfy.utils.ProgressBar(steps)

            from prismaudio_core.inference.sampling import sample_discrete_euler

            def on_step(info):
                pbar.update(1)

            fakes = sample_discrete_euler(
                diffusion.model,
                noise,
                steps,
                callback=on_step,
                **cond_inputs,
                cfg_scale=cfg_scale,
                batch_cfg=True,
            )

            if strategy == "offload_to_cpu":
                diffusion.model.to(get_offload_device())
                diffusion.conditioner.to(get_offload_device())
                soft_empty_cache()
                diffusion.pretransform.to(device)

            # VAE decode in fp32 (snake activations overflow in fp16)
            with torch.amp.autocast(device_type=device.type, enabled=False):
                audio = diffusion.pretransform.decode(fakes.float())

            if strategy == "offload_to_cpu":
                diffusion.pretransform.to(get_offload_device())
                soft_empty_cache()

        # Peak normalize then clamp
        audio = audio.float()
        peak = audio.abs().max().clamp(min=1e-8)
        audio = (audio / peak).clamp(-1, 1)

        return ({"waveform": audio.cpu(), "sample_rate": SAMPLE_RATE},)


# T5-Gemma encoder singleton
_t5_model = None
_t5_tokenizer = None


def _encode_text_t5(text, device, dtype):
    """Encode text using T5-Gemma.

    Uses AutoModelForSeq2SeqLM.get_encoder() to match the reference
    FeaturesUtils.encode_t5_text() implementation.
    No truncation applied (matching reference behavior).
    """
    global _t5_model, _t5_tokenizer

    if _t5_model is None:
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
        model_id = "google/t5gemma-l-l-ul2-it"
        token = resolve_hf_token()
        print(f"[PrismAudio] Loading T5-Gemma text encoder: {model_id}")
        _t5_tokenizer = AutoTokenizer.from_pretrained(model_id, token=token)
        _t5_model = AutoModelForSeq2SeqLM.from_pretrained(model_id, token=token).get_encoder()
        _t5_model.eval()

    _t5_model.to(device, dtype=dtype)

    tokens = _t5_tokenizer(
        text,
        return_tensors="pt",
        padding=True,
    ).to(device)

    with torch.no_grad():
        outputs = _t5_model(**tokens)

    # Move T5 off GPU after encoding to save VRAM
    _t5_model.to("cpu")
    soft_empty_cache()

    return outputs.last_hidden_state.squeeze(0)  # [seq_len, dim]
```

**Step 2: Commit**

```bash
git add nodes/text_only.py
git commit -m "feat: PrismAudioTextOnly node with correct T5-Gemma encoding"
```

---

### Task 10: Integration Testing & Polish

**Files:**
- Modify: `nodes/__init__.py` (verify all imports work)
- Modify: `__init__.py` (verify top-level registration)

**Step 1: Verify all node imports resolve**

Run from ComfyUI's Python:
```bash
cd /path/to/ComfyUI
python -c "
import sys
sys.path.insert(0, 'custom_nodes/ComfyUI-PrismAudio')
from nodes import NODE_CLASS_MAPPINGS
print('Registered nodes:', list(NODE_CLASS_MAPPINGS.keys()))
"
```

Expected output:
```
Registered nodes: ['PrismAudioModelLoader', 'PrismAudioFeatureLoader', 'PrismAudioFeatureExtractor', 'PrismAudioSampler', 'PrismAudioTextOnly']
```

**Step 2: Fix any import errors iteratively**

Common issues:
- `prismaudio_core` internal imports may reference wrong module paths
- Missing model submodules in `prismaudio_core/models/`
- flash-attn fallback not properly guarded

**Step 3: Test model loading (requires GPU + model files)**

```bash
python -c "
from prismaudio_core.factory import create_model_from_config
import json
with open('prismaudio_core/configs/prismaudio.json') as f:
    config = json.load(f)
model = create_model_from_config(config)
print('Model created, params:', sum(p.numel() for p in model.parameters()) / 1e6, 'M')
"
```

Expected: `Model created, params: ~518 M`

**Step 4: End-to-end test with pre-computed features**

If you have a `.npz` feature file from the PrismAudio repo's demo data, test the full pipeline in ComfyUI:
1. PrismAudioModelLoader -> PrismAudioFeatureLoader -> PrismAudioSampler -> Preview Audio node

**Step 5: Verify variable duration handling**

Test with multiple durations (5s, 10s, 20s) to ensure the model adapts to different
input shapes and produces audio of the expected length.

**Step 6: Commit**

```bash
git add -A
git commit -m "feat: integration fixes and verification"
```

---

### Task 11: README

**Files:**
- Create: `README.md`

**Step 1: Write README covering:**

- What PrismAudio is (brief, link to paper)
- Installation (clone, pip install requirements, optional extraction env setup)
- Node descriptions with input/output tables
- Example workflows (quality path with FeatureExtractor, quick path with FeatureLoader, text-only)
- HuggingFace authentication (2 methods: `HF_TOKEN` env var, `huggingface-cli login`)
  - Note: hf_token is NOT a node widget for security reasons
  - Which models may be gated (T5-Gemma, potentially Stable Audio VAE)
- Model file sizes: diffusion ~2.7GB, VAE ~2.5GB, synchformer ~950MB
- Extraction env setup via conda environment.yml
- Troubleshooting (VRAM, flash-attn optional, gated models)
- Credits and license

**Step 2: Commit**

```bash
git add README.md
git commit -m "docs: README with installation and usage instructions"
```

---

## Dependency Graph

```
Task 1 (scaffolding)
  ├── Task 2 (core config + factory) ──┐
  │     └── Task 3 (core models) ──────┤
  │           └── Task 4 (core sampling)┤
  │                                     ├── Task 5 (ModelLoader node)
  │                                     ├── Task 6 (FeatureLoader node)
  │                                     ├── Task 7 (FeatureExtractor node)
  │                                     ├── Task 8 (Sampler node)
  │                                     └── Task 9 (TextOnly node)
  └────────────────────────────────────────── Task 10 (Integration)
                                                └── Task 11 (README)
```

Tasks 5-9 can be parallelized after Task 4 is complete. Task 3 is the heaviest — it involves extracting and adapting ~10 model files.
