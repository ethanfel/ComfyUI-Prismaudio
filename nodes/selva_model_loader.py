import os
from pathlib import Path
import torch
import folder_paths

from .utils import PRISMAUDIO_CATEGORY, get_offload_device, determine_offload_strategy

# Variant → (generator filename, mode, has_bigvgan)
_VARIANTS = {
    "small_16k":  ("generator_small_16k_sup_5.pth",  "16k", True),
    "small_44k":  ("generator_small_44k_sup_5.pth",  "44k", False),
    "medium_44k": ("generator_medium_44k_sup_5.pth", "44k", False),
    "large_44k":  ("generator_large_44k_sup_5.pth",  "44k", False),
}

_SELVA_DIR = Path(folder_paths.models_dir) / "selva"
_PRISMAUDIO_DIR = Path(folder_paths.models_dir) / "prismaudio"


def _ensure(filename, subdir=None):
    """Return path to weight file, downloading it if missing."""
    from selva_core.utils.download_utils import download_model_if_needed
    dest_dir = _SELVA_DIR / subdir if subdir else _SELVA_DIR
    path = dest_dir / filename
    download_model_if_needed(path)
    return str(path)


def _synchformer_path():
    """Return synchformer path, reusing models/prismaudio/ if already present."""
    prismaudio_path = _PRISMAUDIO_DIR / "synchformer_state_dict.pth"
    if prismaudio_path.exists():
        return str(prismaudio_path)
    # Not downloaded for PrismAudio yet — download to models/selva/
    return _ensure("synchformer_state_dict.pth")


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

        print("[SelVA] Resolving weights (auto-downloading if missing)...", flush=True)
        video_enc_path = _ensure("video_enc_sup_5.pth")
        gen_path       = _ensure(gen_filename)
        vae_name = "v1-16.pth" if mode == "16k" else "v1-44.pth"
        vae_path       = _ensure(vae_name, subdir="ext")
        synch_path     = _synchformer_path()
        bigvgan_path   = _ensure("best_netG.pt", subdir="ext") if has_bigvgan else None

        print(f"[SelVA] Loading TextSynch from {video_enc_path}", flush=True)
        net_video_enc = get_my_textsynch("depth1").to(device, dtype).eval()
        net_video_enc.load_weights(
            torch.load(video_enc_path, map_location="cpu", weights_only=False)
        )

        print(f"[SelVA] Loading MMAudio ({variant}) from {gen_path}", flush=True)
        seq_cfg = CONFIG_16K if mode == "16k" else CONFIG_44K
        net_generator = get_my_mmaudio(variant).to(device, dtype).eval()
        net_generator.load_weights(
            torch.load(gen_path, map_location="cpu", weights_only=False)
        )

        print("[SelVA] Loading FeaturesUtils (CLIP + T5 + Synchformer + VAE)...", flush=True)
        feature_utils = FeaturesUtils(
            tod_vae_ckpt=vae_path,
            synchformer_ckpt=synch_path,
            enable_conditions=True,
            mode=mode,
            bigvgan_vocoder_ckpt=bigvgan_path,
            need_vae_encoder=False,
        ).to(device, dtype).eval()

        if strategy == "offload_to_cpu":
            net_generator.to(get_offload_device())
            net_video_enc.to(get_offload_device())
            feature_utils.to(get_offload_device())

        print(f"[SelVA] Model ready: variant={variant} dtype={dtype} strategy={strategy}", flush=True)

        return ({
            "generator":     net_generator,
            "video_enc":     net_video_enc,
            "feature_utils": feature_utils,
            "variant":       variant,
            "mode":          mode,
            "strategy":      strategy,
            "dtype":         dtype,
            "seq_cfg":       seq_cfg,
        },)
