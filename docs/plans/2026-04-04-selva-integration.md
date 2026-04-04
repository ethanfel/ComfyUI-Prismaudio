# SelVA Integration Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add three new ComfyUI nodes (SelvaModelLoader, SelvaFeatureExtractor, SelvaSampler) that run SelVA's text-conditioned V2A pipeline inline — no subprocess, no JAX, pure PyTorch.

**Architecture:** Vendor SelVA source into `selva_core/`, implement three nodes that mirror the PrismAudio pattern. `SelvaFeatureExtractor` takes `SELVA_MODEL` (needs TextSynchformer + CLIP/T5 from FeaturesUtils). `SelvaSampler` runs flow matching ODE with CFG and negative prompts.

**Tech Stack:** PyTorch, open_clip (already in ComfyUI), transformers (already in ComfyUI), torchaudio, einops, torchvision

---

## Design reference

`docs/plans/2026-04-04-selva-integration-design.md`

**Key facts from SelVA source:**
- CLIP input: `[B, T, C, 384, 384]` float32 `[0,1]` — normalization applied inside FeaturesUtils
- Sync input: `[B, T, C, 224, 224]` float32 `[-1,1]` — normalize with `mean=std=[0.5,0.5,0.5]` before passing
- CLIP frame rate: 8fps, Sync frame rate: 25fps
- CONFIG_16K: latent=250, clip=64, sync=192 at 8s
- CONFIG_44K: latent=345, clip=64, sync=192 at 8s
- Sync segments: 16-frame windows, 8-frame stride (overlapping, unlike PrismAudio's 8-frame non-overlapping)
- `net_generator.update_seq_lengths(latent_seq_len, clip_seq_len, sync_seq_len)` must be called before each generation when duration ≠ 8s

---

## Task 1: Create branch and vendor selva_core

**Files:**
- Create: `selva_core/` (full directory tree)

**Step 1: Create new branch off master (not off feature/lora-trainer)**

```bash
git checkout master
git checkout -b feature/selva-integration
```

**Step 2: Clone SelVA and copy source**

```bash
git clone https://github.com/jnwnlee/selva.git /tmp/selva_src
cp -r /tmp/selva_src/selva /media/p5/Comfyui-Prismaudio/selva_core
```

**Step 3: Rename all internal imports**

```bash
cd /media/p5/Comfyui-Prismaudio/selva_core
find . -name "*.py" -exec sed -i \
  's/from selva\./from selva_core./g;
   s/import selva\./import selva_core./g' {} \;
```

**Step 4: Record the pinned commit**

```bash
cd /tmp/selva_src && git rev-parse HEAD
# Paste the hash into a comment at the top of selva_core/__init__.py
```

Edit `selva_core/__init__.py` to add at the top:
```python
# Vendored from https://github.com/jnwnlee/selva
# Pinned commit: <PASTE_HASH_HERE>
# Imports rewritten from selva.* → selva_core.*
```

**Step 5: Verify imports work**

```bash
cd /media/p5/Comfyui-Prismaudio
python -c "
from selva_core.model.networks_generator import MMAudio, get_my_mmaudio
from selva_core.model.networks_video_enc import TextSynch, get_my_textsynch
from selva_core.model.utils.features_utils import FeaturesUtils
from selva_core.model.flow_matching import FlowMatching
from selva_core.model.sequence_config import CONFIG_16K, CONFIG_44K, SequenceConfig
print('selva_core imports OK')
print(f'CONFIG_16K: latent={CONFIG_16K.latent_seq_len} clip={CONFIG_16K.clip_seq_len} sync={CONFIG_16K.sync_seq_len}')
print(f'CONFIG_44K: latent={CONFIG_44K.latent_seq_len} clip={CONFIG_44K.clip_seq_len} sync={CONFIG_44K.sync_seq_len}')
"
```

Expected:
```
selva_core imports OK
CONFIG_16K: latent=250 clip=64 sync=192
CONFIG_44K: latent=345 clip=64 sync=192
```

**Step 6: Commit**

```bash
git add selva_core/
git commit -m "chore: vendor selva_core from jnwnlee/selva@<HASH>

Pure PyTorch SelVA source for SelvaModelLoader/FeatureExtractor/Sampler nodes.
Imports rewritten from selva.* to selva_core.*. No training code included."
```

---

## Task 2: Implement SelvaModelLoader

**Files:**
- Create: `nodes/selva_model_loader.py`
- Modify: `nodes/__init__.py`

**Step 1: Create `nodes/selva_model_loader.py`**

```python
import os
import torch
import folder_paths

from .utils import PRISMAUDIO_CATEGORY, get_offload_device, determine_offload_strategy

# Variant → (generator filename, mode, has_bigvgan)
_VARIANTS = {
    "small_16k":   ("generator_small_16k_sup_5.pth",   "16k", True),
    "small_44k":   ("generator_small_44k_sup_5.pth",   "44k", False),
    "medium_44k":  ("generator_medium_44k_sup_5.pth",  "44k", False),
    "large_44k":   ("generator_large_44k_sup_5.pth",   "44k", False),
}

_SELVA_DIR = os.path.join(folder_paths.models_dir, "selva")


def _selva_path(*parts):
    return os.path.join(_SELVA_DIR, *parts)


def _require(path, hint):
    if not os.path.exists(path):
        raise RuntimeError(
            f"[SelVA] Missing: {path}\n{hint}"
        )
    return path


class SelvaModelLoader:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "variant": (list(_VARIANTS.keys()),),
                "precision": (["bf16", "fp16", "fp32"],),
                "offload_strategy": (["auto", "keep_in_vram", "offload_to_cpu"],),
            }
        }

    RETURN_TYPES = ("SELVA_MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "load_model"
    CATEGORY = PRISMAUDIO_CATEGORY

    def load_model(self, variant, precision, offload_strategy):
        from selva_core.model.networks_generator import get_my_mmaudio
        from selva_core.model.networks_video_enc import get_my_textsynch
        from selva_core.model.utils.features_utils import FeaturesUtils
        from selva_core.model.sequence_config import CONFIG_16K, CONFIG_44K

        gen_filename, mode, has_bigvgan = _VARIANTS[variant]

        dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[precision]
        strategy = determine_offload_strategy(offload_strategy)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Resolve weight paths
        video_enc_path = _require(
            _selva_path("video_enc_sup_5.pth"),
            "Download from https://huggingface.co/jnwnlee/selva and place in models/selva/"
        )
        gen_path = _require(
            _selva_path(gen_filename),
            f"Download {gen_filename} from https://huggingface.co/jnwnlee/selva and place in models/selva/"
        )
        vae_path = _require(
            _selva_path("ext", f"v1-{mode}.pth"),
            f"Download v1-{mode}.pth from MMAudio/SelVA release and place in models/selva/ext/"
        )
        synch_path = _require(
            os.path.join(folder_paths.models_dir, "prismaudio", "synchformer_state_dict.pth"),
            "Synchformer checkpoint missing from models/prismaudio/ — download from FunAudioLLM/PrismAudio"
        )
        bigvgan_path = None
        if has_bigvgan:
            bigvgan_path = _require(
                _selva_path("ext", "best_netG.pt"),
                "Download best_netG.pt (BigVGAN 16k vocoder) from MMAudio release and place in models/selva/ext/"
            )

        print(f"[SelVA] Loading TextSynch from {video_enc_path}", flush=True)
        net_video_enc = get_my_textsynch("depth1").to(device, dtype).eval()
        net_video_enc.load_weights(
            torch.load(video_enc_path, map_location="cpu", weights_only=True)
        )

        print(f"[SelVA] Loading MMAudio ({variant}) from {gen_path}", flush=True)
        seq_cfg = CONFIG_16K if mode == "16k" else CONFIG_44K
        net_generator = get_my_mmaudio(variant).to(device, dtype).eval()
        net_generator.load_weights(
            torch.load(gen_path, map_location="cpu", weights_only=True)
        )

        print(f"[SelVA] Loading FeaturesUtils (CLIP + T5 + Synchformer + VAE)...", flush=True)
        feature_utils = FeaturesUtils(
            tod_vae_ckpt=vae_path,
            synchformer_ckpt=synch_path,
            enable_conditions=True,
            mode=mode,
            bigvgan_vocoder_ckpt=bigvgan_path,
        ).to(device, dtype).eval()

        if strategy == "offload_to_cpu":
            net_generator.to(get_offload_device())
            net_video_enc.to(get_offload_device())
            feature_utils.to(get_offload_device())

        print(f"[SelVA] Model ready: variant={variant} dtype={dtype} strategy={strategy}", flush=True)

        return ({
            "generator": net_generator,
            "video_enc": net_video_enc,
            "feature_utils": feature_utils,
            "variant": variant,
            "mode": mode,
            "strategy": strategy,
            "dtype": dtype,
            "seq_cfg": seq_cfg,
        },)
```

**Step 2: Register in `nodes/__init__.py`**

In the `NODE_CLASS_MAPPINGS` dict, add:
```python
"SelvaModelLoader": (".selva_model_loader", "SelvaModelLoader", "SelVA Model Loader"),
```

**Step 3: Verify node registers**

```bash
cd /media/p5/Comfyui-Prismaudio
python -c "
import sys; sys.path.insert(0, '.')
from nodes.selva_model_loader import SelvaModelLoader
print('inputs:', list(SelvaModelLoader.INPUT_TYPES()['required'].keys()))
print('outputs:', SelvaModelLoader.RETURN_TYPES)
"
```

Expected: `inputs: ['variant', 'precision', 'offload_strategy']`

**Step 4: Commit**

```bash
git add nodes/selva_model_loader.py nodes/__init__.py
git commit -m "feat: SelvaModelLoader node — loads TextSynch + MMAudio + FeaturesUtils"
```

---

## Task 3: Implement SelvaFeatureExtractor

**Files:**
- Create: `nodes/selva_feature_extractor.py`
- Modify: `nodes/__init__.py`

**Step 1: Create `nodes/selva_feature_extractor.py`**

```python
import os
import hashlib
import tempfile

import torch
import torch.nn.functional as F
import numpy as np

from .utils import PRISMAUDIO_CATEGORY, get_device, get_offload_device, soft_empty_cache

# SelVA video preprocessing constants (from selva/utils/eval_utils.py)
_CLIP_SIZE = 384
_SYNC_SIZE = 224
_CLIP_FPS  = 8
_SYNC_FPS  = 25

# Sync normalization: [-1, 1] (from selva/utils/eval_utils.py load_video)
_SYNC_MEAN = torch.tensor([0.5, 0.5, 0.5]).view(1, 3, 1, 1)
_SYNC_STD  = torch.tensor([0.5, 0.5, 0.5]).view(1, 3, 1, 1)


def _sample_frames(video, source_fps, target_fps, duration):
    """Sample frames from [T,H,W,C] float32 [0,1] at target_fps."""
    T = video.shape[0]
    n_out = max(1, int(duration * target_fps))
    indices = [min(int(i / target_fps * source_fps), T - 1) for i in range(n_out)]
    return video[indices]  # [N, H, W, C]


def _resize_frames(frames, size):
    """Resize [N,H,W,C] float32 [0,1] → [N,C,H,W] at target size."""
    x = frames.permute(0, 3, 1, 2)  # [N, C, H, W]
    x = F.interpolate(x, size=(size, size), mode="bicubic", align_corners=False)
    return x.clamp(0, 1)  # [N, C, H, W] float32


def _hash_inputs(video_tensor, prompt, fps, variant):
    h = hashlib.sha256()
    h.update(video_tensor.cpu().numpy().tobytes()[:1024 * 1024])
    h.update(prompt.encode())
    h.update(str(fps).encode())
    h.update(variant.encode())
    return h.hexdigest()[:16]


class SelvaFeatureExtractor:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model":   ("SELVA_MODEL",),
                "video":   ("IMAGE",),
                "prompt":  ("STRING", {"default": "", "multiline": True,
                                       "tooltip": "Text prompt used by TextSynchformer to focus sync features on the relevant sound source. Should match the prompt used in SelvaSampler."}),
            },
            "optional": {
                "video_info": ("VHS_VIDEOINFO", {"tooltip": "Connect VHS LoadVideo info to auto-set fps."}),
                "fps":   ("FLOAT", {"default": 30.0, "min": 1.0, "max": 120.0, "step": 0.001}),
                "duration": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 30.0, "step": 0.1,
                                       "tooltip": "Override duration in seconds. 0 = infer from video length and fps."}),
                "cache_dir": ("STRING", {"default": "", "tooltip": "Directory for cached .npz features. Empty = temp dir."}),
            },
        }

    RETURN_TYPES = ("SELVA_FEATURES", "FLOAT")
    RETURN_NAMES = ("features", "fps")
    FUNCTION = "extract_features"
    CATEGORY = PRISMAUDIO_CATEGORY

    def extract_features(self, model, video, prompt, video_info=None, fps=30.0,
                         duration=0.0, cache_dir=""):
        if video_info is not None:
            fps = video_info["loaded_fps"]

        T = video.shape[0]
        if duration <= 0:
            duration = T / fps
        duration = min(duration, T / fps)  # clamp to actual video length

        if not prompt.strip():
            print("[SelVA] Warning: empty prompt — TextSynchformer sync features will be unfocused.", flush=True)

        # Cache
        if not cache_dir:
            cache_dir = os.path.join(tempfile.gettempdir(), "selva_features")
        os.makedirs(cache_dir, exist_ok=True)
        cache_key = _hash_inputs(video, prompt, fps, model["variant"])
        cached_path = os.path.join(cache_dir, f"{cache_key}.npz")

        if os.path.exists(cached_path):
            print(f"[SelVA] Using cached features: {cached_path}", flush=True)
            return (_load_cached(cached_path), float(fps))

        device = get_device()
        dtype  = model["dtype"]
        strategy = model["strategy"]
        feature_utils = model["feature_utils"]
        net_video_enc = model["video_enc"]

        # Move feature models to device
        if strategy == "offload_to_cpu":
            feature_utils.to(device)
            net_video_enc.to(device)
            soft_empty_cache()

        print(f"[SelVA] Extracting features: duration={duration:.2f}s fps={fps:.3f} prompt='{prompt[:60]}'", flush=True)

        with torch.no_grad():
            # --- CLIP frames: 384×384, [0,1], 8fps ---
            clip_frames = _sample_frames(video, fps, _CLIP_FPS, duration)  # [N, H, W, C]
            clip_frames = _resize_frames(clip_frames, _CLIP_SIZE)          # [N, C, 384, 384]
            clip_input  = clip_frames.unsqueeze(0).to(device, dtype)       # [1, N, C, 384, 384]
            print(f"[SelVA]   CLIP frames: {clip_frames.shape[0]} @ {_CLIP_FPS}fps", flush=True)

            clip_features = feature_utils.encode_video_with_clip(clip_input)  # [1, N, 1024]

            # --- Sync frames: 224×224, [-1,1], 25fps ---
            n_sync = max(16, int(duration * _SYNC_FPS))  # minimum 16 for segmentation
            sync_frames = _sample_frames(video, fps, _SYNC_FPS, duration)
            if sync_frames.shape[0] < 16:
                # Pad by repeating last frame to reach minimum 16
                pad = 16 - sync_frames.shape[0]
                sync_frames = torch.cat([sync_frames, sync_frames[-1:].expand(pad, -1, -1, -1)], dim=0)
            sync_frames = _resize_frames(sync_frames, _SYNC_SIZE)          # [N, C, 224, 224]
            # Normalize to [-1, 1]
            mean = _SYNC_MEAN.to(sync_frames.device)
            std  = _SYNC_STD.to(sync_frames.device)
            sync_frames = (sync_frames - mean) / std
            sync_input  = sync_frames.unsqueeze(0).to(device, dtype)       # [1, N, C, 224, 224]
            print(f"[SelVA]   Sync frames: {sync_frames.shape[0]} @ {_SYNC_FPS}fps", flush=True)

            # Encode T5 text + prepend supplementary tokens → text-conditioned sync features
            text_f_t5, text_mask = feature_utils.encode_text_t5([prompt])  # [1, L, 768], [1, L]
            text_f_t5, text_mask = net_video_enc.prepend_sup_text_tokens(text_f_t5, text_mask)
            sync_features = net_video_enc.encode_video_with_sync(
                sync_input, text_f=text_f_t5, text_mask=text_mask
            )  # [1, T_sync, 768]

        print(f"[SelVA]   clip_features: {tuple(clip_features.shape)}", flush=True)
        print(f"[SelVA]   sync_features: {tuple(sync_features.shape)}", flush=True)

        # Offload back if needed
        if strategy == "offload_to_cpu":
            feature_utils.to(get_offload_device())
            net_video_enc.to(get_offload_device())
            soft_empty_cache()

        # Save cache
        np.savez(
            cached_path,
            clip_features=clip_features.cpu().float().numpy(),
            sync_features=sync_features.cpu().float().numpy(),
            duration=duration,
        )
        print(f"[SelVA] Features cached: {cached_path}", flush=True)

        features = {
            "clip_features": clip_features.cpu(),
            "sync_features": sync_features.cpu(),
            "duration": duration,
        }
        return (features, float(fps))


def _load_cached(path):
    data = np.load(path, allow_pickle=False)
    return {
        "clip_features": torch.from_numpy(data["clip_features"]),
        "sync_features": torch.from_numpy(data["sync_features"]),
        "duration": float(data["duration"]),
    }
```

**Step 2: Register in `nodes/__init__.py`**

```python
"SelvaFeatureExtractor": (".selva_feature_extractor", "SelvaFeatureExtractor", "SelVA Feature Extractor"),
```

**Step 3: Verify node registers**

```bash
python -c "
import sys; sys.path.insert(0, '.')
from nodes.selva_feature_extractor import SelvaFeatureExtractor
inputs = SelvaFeatureExtractor.INPUT_TYPES()
print('required:', list(inputs['required'].keys()))
print('optional:', list(inputs['optional'].keys()))
print('outputs:', SelvaFeatureExtractor.RETURN_TYPES)
"
```

Expected: `required: ['model', 'video', 'prompt']`

**Step 4: Commit**

```bash
git add nodes/selva_feature_extractor.py nodes/__init__.py
git commit -m "feat: SelvaFeatureExtractor — inline CLIP + TextSynchformer feature extraction"
```

---

## Task 4: Implement SelvaSampler

**Files:**
- Create: `nodes/selva_sampler.py`
- Modify: `nodes/__init__.py`

**Step 1: Create `nodes/selva_sampler.py`**

```python
import math
import torch
import comfy.utils

from .utils import (
    PRISMAUDIO_CATEGORY,
    get_device, get_offload_device, soft_empty_cache,
)


def _make_seq_cfg(duration, mode):
    """Compute sequence lengths for a given duration and mode."""
    from selva_core.model.sequence_config import SequenceConfig
    if mode == "16k":
        return SequenceConfig(duration=duration, sampling_rate=16000, spectrogram_frame_rate=256)
    else:
        return SequenceConfig(duration=duration, sampling_rate=44100, spectrogram_frame_rate=512)


class SelvaSampler:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model":    ("SELVA_MODEL",),
                "features": ("SELVA_FEATURES",),
                "prompt":   ("STRING", {"default": "", "multiline": True,
                                        "tooltip": "Should match the prompt used in SelvaFeatureExtractor."}),
                "negative_prompt": ("STRING", {"default": "", "multiline": True,
                                               "tooltip": "Sounds to steer away from, e.g. 'wind noise, background music'."}),
                "duration": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 30.0, "step": 0.1,
                                       "tooltip": "Audio duration in seconds. 0 = use duration from features."}),
                "steps":    ("INT",   {"default": 25,  "min": 1,   "max": 200}),
                "cfg_strength": ("FLOAT", {"default": 4.5, "min": 1.0, "max": 20.0, "step": 0.1}),
                "seed":     ("INT",   {"default": 0,   "min": 0,   "max": 0xFFFFFFFF}),
            },
        }

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "generate"
    CATEGORY = PRISMAUDIO_CATEGORY

    def generate(self, model, features, prompt, negative_prompt, duration, steps, cfg_strength, seed):
        from selva_core.model.flow_matching import FlowMatching

        device   = get_device()
        dtype    = model["dtype"]
        strategy = model["strategy"]
        net_generator = model["generator"]
        feature_utils = model["feature_utils"]
        mode          = model["mode"]

        # Resolve duration
        if duration <= 0:
            if "duration" not in features:
                raise ValueError("[SelVA] duration=0 but features contain no duration field.")
            duration = features["duration"]
            print(f"[SelVA] Using video duration from features: {duration:.2f}s", flush=True)

        seq_cfg = _make_seq_cfg(duration, mode)
        sample_rate = seq_cfg.sampling_rate

        # Move models to device
        if strategy == "offload_to_cpu":
            net_generator.to(device)
            feature_utils.to(device)
            soft_empty_cache()

        clip_f = features["clip_features"].to(device, dtype)  # [1, T_clip, 1024]
        sync_f = features["sync_features"].to(device, dtype)  # [1, T_sync, 768]

        print(f"[SelVA] clip_f={tuple(clip_f.shape)} sync_f={tuple(sync_f.shape)}", flush=True)
        print(f"[SelVA] seq_cfg: latent={seq_cfg.latent_seq_len} clip={seq_cfg.clip_seq_len} sync={seq_cfg.sync_seq_len}", flush=True)

        # Update model sequence lengths for this duration
        net_generator.update_seq_lengths(
            latent_seq_len=seq_cfg.latent_seq_len,
            clip_seq_len=seq_cfg.clip_seq_len,
            sync_seq_len=seq_cfg.sync_seq_len,
        )

        with torch.no_grad():
            # Encode text
            text_clip = feature_utils.encode_text_clip([prompt])    # [1, 77, D]

            # Build empty (negative) conditions for CFG
            neg_text_clip = feature_utils.encode_text_clip([negative_prompt]) \
                if negative_prompt.strip() else None

            conditions = net_generator.preprocess_conditions(clip_f, sync_f, text_clip)
            empty_conditions = net_generator.get_empty_conditions(
                bs=1, negative_text_features=neg_text_clip
            )

            # Sample initial noise
            rng = torch.Generator(device=device).manual_seed(seed)
            x0 = torch.randn(
                1, seq_cfg.latent_seq_len, net_generator.latent_dim,
                device=device, dtype=dtype, generator=rng
            )

            # Flow matching ODE (Euler)
            fm = FlowMatching(min_sigma=0, inference_mode="euler", num_steps=steps)
            pbar = comfy.utils.ProgressBar(steps)

            _step_count = [0]
            orig_to_data = fm.to_data

            def tracked_to_data(fn, x0_):
                # ProgressBar update via step counting in ode_wrapper
                return orig_to_data(fn, x0_)

            # Wrap ODE to update progress bar
            def ode_wrapper_tracked(t, x):
                _step_count[0] += 1
                pbar.update(1)
                return net_generator.ode_wrapper(t, x, conditions, empty_conditions, cfg_strength)

            x1 = fm.to_data(ode_wrapper_tracked, x0)

        print(f"[SelVA] latent stats: mean={x1.float().mean():.4f} std={x1.float().std():.4f}", flush=True)

        # Decode: latent → mel → audio
        if strategy == "offload_to_cpu":
            feature_utils.to(device)
            soft_empty_cache()

        with torch.no_grad():
            x1_unnorm = net_generator.unnormalize(x1)
            spec  = feature_utils.decode(x1_unnorm)
            audio = feature_utils.vocode(spec)   # [1, samples] or [1, 1, samples]

        if strategy == "offload_to_cpu":
            net_generator.to(get_offload_device())
            feature_utils.to(get_offload_device())
            soft_empty_cache()

        # Normalise to [-1, 1]
        audio = audio.float()
        if audio.dim() == 2:
            audio = audio.unsqueeze(1)   # [1, 1, samples]
        elif audio.dim() == 3 and audio.shape[1] != 1:
            audio = audio.mean(dim=1, keepdim=True)  # stereo → mono

        peak = audio.abs().max().clamp(min=1e-8)
        audio = (audio / peak).clamp(-1, 1)
        print(f"[SelVA] audio: shape={tuple(audio.shape)} sr={sample_rate}", flush=True)

        return ({"waveform": audio.cpu(), "sample_rate": sample_rate},)
```

**Step 2: Register in `nodes/__init__.py`**

```python
"SelvaSampler": (".selva_sampler", "SelvaSampler", "SelVA Sampler"),
```

**Step 3: Verify node registers**

```bash
python -c "
import sys; sys.path.insert(0, '.')
from nodes.selva_sampler import SelvaSampler
inputs = SelvaSampler.INPUT_TYPES()
print('inputs:', list(inputs['required'].keys()))
print('outputs:', SelvaSampler.RETURN_TYPES)
"
```

Expected: `inputs: ['model', 'features', 'prompt', 'negative_prompt', 'duration', 'steps', 'cfg_strength', 'seed']`

**Step 4: Commit**

```bash
git add nodes/selva_sampler.py nodes/__init__.py
git commit -m "feat: SelvaSampler — flow matching ODE with CFG + negative prompts"
```

---

## Task 5: Create example workflow and push

**Files:**
- Create: `workflows/selva_video_to_audio.json`

**Step 1: Create workflow JSON**

Create `workflows/selva_video_to_audio.json` with this node graph:
- LoadVideo (VHS) → IMAGE + VHS_VIDEOINFO
- SelvaModelLoader → SELVA_MODEL
- SelvaFeatureExtractor (takes IMAGE + VHS_VIDEOINFO + SELVA_MODEL, prompt) → SELVA_FEATURES
- SelvaSampler (takes SELVA_MODEL + SELVA_FEATURES, prompt, negative_prompt) → AUDIO
- PreviewAudio (takes AUDIO)

Set defaults: variant=medium_44k, precision=bf16, steps=25, cfg_strength=4.5, duration=0.

**Step 2: Push branch**

```bash
git push -u origin feature/selva-integration
```

---

## Task 6: Smoke test

**Step 1: Check all three nodes are importable from ComfyUI's perspective**

```bash
cd /media/p5/Comfyui-Prismaudio
python -c "
import sys; sys.path.insert(0, '.')
import nodes
m = nodes.NODE_CLASS_MAPPINGS
print('SelVA nodes:', [k for k in m if 'Selva' in k])
assert 'SelvaModelLoader' in m
assert 'SelvaFeatureExtractor' in m
assert 'SelvaSampler' in m
print('All SelVA nodes registered OK')
"
```

**Step 2: Verify no import errors in full node load**

```bash
python -c "
import sys; sys.path.insert(0, '.')
from nodes.selva_model_loader import SelvaModelLoader
from nodes.selva_feature_extractor import SelvaFeatureExtractor
from nodes.selva_sampler import SelvaSampler
print('All imports clean')
"
```

**Step 3: Final commit with any fixes**

```bash
git add -A
git commit -m "fix: selva integration smoke test fixes (if any)"
git push
```

---

## Notes

- The `FeaturesUtils.train()` is overridden to always call `super().train(False)` — SelVA models are always in eval mode
- `net_generator.update_seq_lengths` recalculates rotary position embeddings; call it before every generation when duration may vary
- ProgressBar tracking: `FlowMatching.to_data` calls `fn(t, x)` for each Euler step; wrapping `ode_wrapper` with a counter gives accurate progress
- The `feature_utils.vocode` returns audio at 16kHz for small_16k (uses BigVGAN) and 44.1kHz for 44k variants (uses VAE mel decoder directly)
- If `encode_text_t5` or `encode_text_clip` fail with missing model errors on first run, it's HuggingFace downloading `flan-t5-base` and `apple/DFN5B-CLIP-ViT-H-14-384` — this is expected and takes a few minutes once
