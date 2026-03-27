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
