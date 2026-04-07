import json
from pathlib import Path

import folder_paths

from .utils import SELVA_CATEGORY


class SelvaDatasetBrowser:
    """Browse a dataset.json file entry by entry using an integer index.

    Each entry in the JSON is expected to have:
      - "path"  : base path (no extension) — directory that holds frame images
      - "label" : text description of the clip

    Derived outputs:
      - video_path  : path + ".mp4"
      - audio_path  : path + ".wav"
      - frames_dir  : path  (the directory itself, for image-sequence loaders)
      - label       : entry["label"]
      - count       : total number of entries in the file
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "dataset_json": ("STRING", {
                    "default": "",
                    "tooltip": "Absolute or ComfyUI-relative path to a dataset.json file.",
                }),
                "index": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 9999,
                    "step": 1,
                    "tooltip": "Zero-based index of the entry to inspect.",
                }),
            },
        }

    RETURN_TYPES  = ("STRING", "STRING", "STRING", "STRING", "STRING", "STRING", "STRING", "INT")
    RETURN_NAMES  = ("video_path", "audio_wav", "audio_flac", "features_path", "frames_dir", "mask_dir", "label", "max_index")
    OUTPUT_TOOLTIPS = (
        "path + '.mp4'",
        "features/ + name + '.wav'",
        "features/ + name + '.flac'",
        "features/ + name + '.npz'  (pre-extracted SelVA features)",
        "path  (image-sequence directory)",
        "path + '_mask'  (mask image-sequence directory)",
        "Text label for this clip",
        "count - 1 — wire to a primitive INT's max to constrain the index widget",
    )
    FUNCTION  = "browse"
    CATEGORY  = SELVA_CATEGORY
    DESCRIPTION = (
        "Reads a dataset.json produced by the SelVA dataset preparation pipeline "
        "and exposes one entry at a time via an integer index. "
        "Outputs the video path, audio path, frames directory, label, and total entry count."
    )

    # Re-read the file every call so edits are picked up without restarting ComfyUI.
    IS_CHANGED = classmethod(lambda cls, **_: float("nan"))

    def browse(self, dataset_json: str, index: int):
        p = Path(dataset_json.strip())
        if not p.is_absolute():
            p = Path(folder_paths.base_path) / p
        if not p.exists():
            raise FileNotFoundError(f"[SelVA Dataset Browser] File not found: {p}")

        with p.open("r", encoding="utf-8") as f:
            data = json.load(f)

        if not isinstance(data, list) or len(data) == 0:
            raise ValueError(f"[SelVA Dataset Browser] Expected a non-empty JSON array in {p}")

        count = len(data)
        if index >= count:
            raise IndexError(
                f"[SelVA Dataset Browser] index {index} is out of range "
                f"(dataset has {count} entries, last index is {count - 1})"
            )
        entry = data[index]

        base  = entry["path"]
        label = entry.get("label", "")

        p_base    = Path(base)
        feat_base = str(p_base.parent / "features" / p_base.name)

        print(
            f"[SelVA Dataset Browser] {index + 1}/{count}  label='{label}'  base={base}",
            flush=True,
        )

        return (
            base + ".mp4",
            feat_base + ".wav",
            feat_base + ".flac",
            feat_base + ".npz",
            base,
            base + "_mask",
            label,
            count - 1,
        )
