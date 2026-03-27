"""
ComfyUI-PrismAudio: Video-to-Audio and Text-to-Audio generation using PrismAudio (ICLR 2026).
"""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
