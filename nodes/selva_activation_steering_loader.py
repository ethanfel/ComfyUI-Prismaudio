"""SelVA Activation Steering Loader.

Loads a steering_vectors.pt bundle produced by SelVA Activation Steering Extractor
and returns a STEERING_VECTORS dict for use by SelVA Sampler.
"""

from pathlib import Path

import torch
import folder_paths

from .utils import SELVA_CATEGORY


class SelvaActivationSteeringLoader:
    CATEGORY    = SELVA_CATEGORY
    FUNCTION    = "load"
    RETURN_TYPES  = ("STEERING_VECTORS",)
    RETURN_NAMES  = ("steering_vectors",)
    OUTPUT_TOOLTIPS = ("Steering vectors bundle — connect to SelVA Sampler's steering_vectors input.",)
    DESCRIPTION = (
        "Loads activation steering vectors from a .pt file produced by "
        "SelVA Activation Steering Extractor. Connect to SelVA Sampler to nudge "
        "denoising toward the target activation patterns."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "path": ("STRING", {
                    "default": "steering_vectors.pt",
                    "tooltip": "Path to steering_vectors.pt. Relative paths resolve to ComfyUI output directory.",
                }),
            },
        }

    def load(self, path):
        p = Path(path.strip())
        if not p.is_absolute():
            p = Path(folder_paths.get_output_directory()) / p
        if not p.exists():
            raise FileNotFoundError(f"[Steering] File not found: {p}")

        payload = torch.load(str(p), map_location="cpu", weights_only=False)

        n_blocks = payload["n_blocks"]
        n_joint  = payload["n_joint"]
        n_fused  = payload["n_fused"]
        n_vecs   = len(payload["steering_vectors"])

        print(f"[Steering] Loaded: {p}", flush=True)
        print(f"[Steering] blocks={n_blocks} (joint={n_joint} fused={n_fused})  "
              f"latent_seq_len={payload['latent_seq_len']}  "
              f"n_samples={payload['n_samples']}", flush=True)
        print(f"[Steering] mode={payload.get('mode')}  variant={payload.get('variant')}", flush=True)

        norms = [payload["steering_vectors"][i].norm().item() for i in range(n_vecs)]
        mean_norm = sum(norms) / len(norms)
        print(f"[Steering] Mean steering norm across {n_vecs} blocks: {mean_norm:.4f}", flush=True)

        return (payload,)
