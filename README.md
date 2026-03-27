# ComfyUI-PrismAudio

Custom nodes for [PrismAudio](https://github.com/FunAudioLLM/ThinkSound) (ICLR 2026) — video-to-audio and text-to-audio generation using decomposed Chain-of-Thought reasoning with a 518M parameter DiT diffusion model and Stable Audio 2.0 VAE.

## Installation

Clone into your ComfyUI custom nodes directory:

```bash
cd ComfyUI/custom_nodes
git clone -b prismaudio https://github.com/FunAudioLLM/ThinkSound ComfyUI-PrismAudio
pip install -r ComfyUI-PrismAudio/requirements.txt
```

**flash-attn** is optional. It is detected at runtime and falls back to PyTorch SDPA if unavailable.

For the **Feature Extractor** node (video feature extraction), a separate conda environment is required — see [Feature Extraction Environment](#feature-extraction-environment) below.

## Nodes

| Node | Description |
|------|-------------|
| **PrismAudio Model Loader** | Loads the diffusion model and VAE. Auto-downloads weights from HuggingFace. Inputs: `precision` (auto/fp32/fp16/bf16), `offload_strategy` (auto/keep_in_vram/offload_to_cpu). |
| **PrismAudio Feature Loader** | Loads pre-computed `.npz` feature files for use with the sampler. |
| **PrismAudio Feature Extractor** | Subprocess bridge that extracts features from video. Requires a separate conda env with JAX/TF. |
| **PrismAudio Sampler** | Main generation node. Takes model + features, produces AUDIO. Inputs: `duration`, `steps`, `cfg_scale`, `seed`. |
| **PrismAudio Text Only** | Text-to-audio generation without video. Uses the T5-Gemma text encoder. Inputs: `text_prompt`, `duration`, `steps`, `cfg_scale`, `seed`. |

## Workflows

### Quality Path (Video-to-Audio)

```
Video → PrismAudio Feature Extractor → PrismAudio Sampler → Save Audio
```

### Pre-computed Path

```
PrismAudio Feature Loader (.npz) → PrismAudio Sampler → Save Audio
```

### Text-Only

```
PrismAudio Text Only → Save Audio
```

> **Note:** CoT text is a STRING input on the sampler. You can use any existing ComfyUI LLM nodes to generate it.

## HuggingFace Authentication

Required for gated models (T5-Gemma, and possibly Stable Audio VAE).

1. Visit <https://huggingface.co/FunAudioLLM/PrismAudio> and accept the license.
2. Authenticate via one of:
   - **Environment variable:** `export HF_TOKEN=hf_...`
   - **CLI login:** `huggingface-cli login`

There is no `hf_token` widget on the nodes by design — ComfyUI saves all STRING values to workflow JSON, which would expose your token.

## Model Files

Weights are auto-downloaded to `ComfyUI/models/prismaudio/`:

| File | Size | Description |
|------|------|-------------|
| `prismaudio.ckpt` | ~2.7 GB | Diffusion model |
| `vae.ckpt` | ~2.5 GB | Stable Audio 2.0 VAE |
| `synchformer_state_dict.pth` | ~950 MB | Synchformer |

T5-Gemma is cached in the standard HuggingFace cache directory (`~/.cache/huggingface/`).

## VRAM Requirements

| VRAM | Strategy |
|------|----------|
| 24 GB+ | Keep all models in VRAM |
| 12–24 GB | Sequential offload |
| 8–12 GB | Aggressive offload + fp16 |
| < 8 GB | May work with aggressive offload |

## Feature Extraction Environment

The **PrismAudio Feature Extractor** node runs extraction in a subprocess using a separate Python environment (JAX/TF dependencies).

```bash
conda env create -f scripts/environment.yml
conda activate prismaudio-extract
```

Then set the `python_env` input on the Feature Extractor node to:

```
/path/to/conda/envs/prismaudio-extract/bin/python
```

## Troubleshooting

- **Gated model errors** — Accept the license at <https://huggingface.co/FunAudioLLM/PrismAudio> and set `HF_TOKEN`.
- **VRAM errors** — Switch `offload_strategy` to `offload_to_cpu`, or use `fp16` precision.
- **flash-attn** — Purely optional. Auto-detected at runtime; falls back to PyTorch SDPA.

## Credits

PrismAudio by [FunAudioLLM](https://github.com/FunAudioLLM) (ICLR 2026). [Paper & code](https://github.com/FunAudioLLM/ThinkSound).
