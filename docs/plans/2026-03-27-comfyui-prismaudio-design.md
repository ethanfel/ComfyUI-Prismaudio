# ComfyUI-PrismAudio Design Document

**Date:** 2026-03-27
**Status:** Approved

## Overview

ComfyUI nodes for PrismAudio (ICLR 2026) — video-to-audio and text-to-audio generation. PrismAudio uses decomposed Chain-of-Thought reasoning across 4 dimensions (Semantic, Temporal, Aesthetic, Spatial) with a 518M parameter DiT diffusion model and Stable Audio 2.0 VAE.

## Architecture

**Approach C: Selective Code Extraction** — Extract only inference-critical code from PrismAudio into a self-contained `prismaudio_core/` module. No JAX/TensorFlow in the ComfyUI environment. Feature extraction via separate isolated environment.

## Project Structure

```
ComfyUI-PrismAudio/
├── __init__.py                    # Node registration
├── nodes/
│   ├── __init__.py
│   ├── model_loader.py            # PrismAudioModelLoader
│   ├── feature_loader.py          # PrismAudioFeatureLoader (loads .npz)
│   ├── feature_extractor.py       # PrismAudioFeatureExtractor (subprocess bridge)
│   ├── sampler.py                 # PrismAudioSampler
│   ├── text_only.py               # PrismAudioTextOnly
│   └── utils.py                   # Shared helpers
├── prismaudio_core/               # Extracted inference code from PrismAudio
│   ├── __init__.py
│   ├── configs/
│   │   └── prismaudio.json
│   ├── models/                    # DiT, conditioners, autoencoders, etc.
│   ├── inference/                 # sampling.py, generation.py
│   └── factory.py                 # create_model_from_config
├── scripts/
│   ├── extract_features.py        # Standalone VideoPrism feature extraction
│   └── environment.yml            # Conda env for extraction (JAX + TF)
├── requirements.txt               # PyTorch-only deps (no JAX/TF)
└── README.md
```

## Nodes

### PrismAudioModelLoader

Loads the diffusion model + VAE. Auto-downloads from HuggingFace if weights not found locally.

| Field | Type | Details |
|-------|------|---------|
| **Inputs** | | |
| precision | COMBO | [auto, fp32, fp16, bf16] — auto detects GPU capability |
| offload_strategy | COMBO | [auto, keep_in_vram, offload_to_cpu] |
| *(no hf_token widget — security risk, would be saved to workflow JSON)* | | |
| **Output** | | |
| model | PRISMAUDIO_MODEL | Dict containing diffusion model + VAE + config |

**Token resolution order** (no widget — env/CLI only for security):
1. `HF_TOKEN` environment variable
2. `huggingface-cli login` cached token
3. None — fails on gated models with clear error message linking to license page

**Auto-download:** Uses `huggingface_hub.hf_hub_download()` from `FunAudioLLM/PrismAudio`. Models stored in `ComfyUI/models/prismaudio/`. Users can also place files manually.

### PrismAudioFeatureLoader

Loads pre-computed `.npz` feature files for maximum quality video-to-audio.

| Field | Type | Details |
|-------|------|---------|
| **Inputs** | | |
| npz_path | STRING | Path to .npz file |
| **Output** | | |
| features | PRISMAUDIO_FEATURES | Dict with video_features, global_video_features, text_features, global_text_features, sync_features |

### PrismAudioFeatureExtractor

Subprocess bridge — extracts features from video using VideoPrism in an isolated environment.

| Field | Type | Details |
|-------|------|---------|
| **Inputs** | | |
| video | IMAGE | ComfyUI video frames tensor |
| caption_cot | STRING | CoT description text |
| python_env | STRING | Path to python binary with JAX/TF (default: "python") |
| output_dir | STRING | Cache directory for .npz files (default: temp dir) |
| **Output** | | |
| features | PRISMAUDIO_FEATURES | Same format as FeatureLoader output |

**Caching:** Hashes video + text to avoid re-extraction on repeated runs.

### PrismAudioSampler

Main generation node — takes model + features, produces audio.

| Field | Type | Details |
|-------|------|---------|
| **Inputs** | | |
| model | PRISMAUDIO_MODEL | From ModelLoader |
| features | PRISMAUDIO_FEATURES | From FeatureLoader or FeatureExtractor |
| cot_description | STRING | Multiline CoT text |
| duration | FLOAT | 1.0-30.0, defaults to video length |
| steps | INT | 1-100, default 24 |
| cfg_scale | FLOAT | 1.0-20.0, default 5.0 |
| seed | INT | Controls noise generation |
| **Output** | | |
| audio | AUDIO | {waveform: tensor, sample_rate: 44100} |

**Pipeline:**
1. Encode CoT text via T5-Gemma -> text_features
2. Assemble conditioning (cross_attn_cond, add_cond, sync_cond)
3. Compute latent_seq_len = round(44100 / 2048 * duration)
4. Generate noise [1, 64, latent_seq_len] from seed
5. Discrete Euler sampling (rectified flow) with CFG
6. VAE decode -> stereo waveform at 44100 Hz
7. Normalize to [-1, 1], return as AUDIO

### PrismAudioTextOnly

Text-to-audio without video input.

| Field | Type | Details |
|-------|------|---------|
| **Inputs** | | |
| model | PRISMAUDIO_MODEL | From ModelLoader |
| text_prompt | STRING | Text description |
| duration | FLOAT | 1.0-30.0 |
| steps | INT | 1-100, default 24 |
| cfg_scale | FLOAT | 1.0-20.0, default 5.0 |
| seed | INT | Controls noise generation |
| **Output** | | |
| audio | AUDIO | {waveform: tensor, sample_rate: 44100} |

Uses empty tensors for video/sync features, T5-Gemma encodes the text prompt.

## VRAM Management

Adaptive strategy using `comfy.model_management`:

| Available VRAM | Behavior |
|---|---|
| 24GB+ | Keep diffusion + VAE in VRAM |
| 12-24GB | Sequential offload between stages |
| 8-12GB | Aggressive offload, one component on GPU at a time, fp16 forced |
| <8GB | Warn user, attempt with aggressive offload + fp16 |

Key APIs: `mm.get_torch_device()`, `mm.get_free_memory()`, `mm.soft_empty_cache()`, `mm.unet_offload_device()`

## Feature Extraction Paths

### Path 1: Pre-computed .npz (FeatureLoader)
User runs `scripts/extract_features.py` externally in the extraction conda env. Loads result into ComfyUI. Original VideoPrism quality, zero ComfyUI env risk.

### Path 2: Subprocess bridge (FeatureExtractor)
Node calls extraction script as subprocess using a user-specified Python binary. Seamless in-ComfyUI experience, JAX runs isolated. Caches results by content hash.

### Path 3: Text-only (TextOnly node)
No video features needed. T5-Gemma text encoding only (PyTorch-native).

## Dependencies

### ComfyUI environment (`requirements.txt`)
```
einops>=0.7.0
safetensors
huggingface_hub
transformers>=4.52.3
k-diffusion>=0.1.1
```

flash-attn: Optional, detected at runtime. Falls back to `torch.nn.functional.scaled_dot_product_attention`.

### Extraction environment (`scripts/environment.yml`)
Separate conda environment with JAX, tensorflow-cpu==2.15.0, VideoPrism, Synchformer, decord. Provided as ready-made conda env file for one-command setup.

## Model Files

Stored in `ComfyUI/models/prismaudio/`:

| File | Size | Source |
|------|------|--------|
| prismaudio.ckpt | ~2GB | FunAudioLLM/PrismAudio |
| vae.ckpt | ~2.5GB | FunAudioLLM/PrismAudio |
| synchformer_state_dict.pth | ~950MB | FunAudioLLM/PrismAudio |

T5-Gemma (`google/t5gemma-l-l-ul2-it`) cached in standard HuggingFace cache.

Registered via: `folder_paths.add_model_folder_path("prismaudio", ...)`

## Design Decisions

- **Composable**: Standard AUDIO output, CoT as plain STRING input. No reinventing save/preview/mux nodes.
- **No JAX/TF in ComfyUI env**: All JAX-dependent code isolated in extraction script/env.
- **LLM-agnostic CoT**: Users bring their own CoT generation via existing LLM nodes — better models available than bundled Qwen2.5-VL.
- **HF token via env/CLI only**: No widget (ComfyUI saves all STRING values to workflow JSON). Uses `HF_TOKEN` env var or `huggingface-cli login`.
- **flash-attn optional**: Avoids installation headaches, uses PyTorch SDPA as fallback.
