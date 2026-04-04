import torch
import comfy.model_management as mm

SELVA_CATEGORY = "SelVA"

def get_device():
    return mm.get_torch_device()

def get_offload_device():
    return mm.unet_offload_device()

def soft_empty_cache():
    mm.soft_empty_cache()

def determine_offload_strategy(preference):
    if preference != "auto":
        return preference
    free_mem = mm.get_free_memory(get_device())
    if free_mem / (1024 ** 3) >= 16:
        return "keep_in_vram"
    return "offload_to_cpu"
