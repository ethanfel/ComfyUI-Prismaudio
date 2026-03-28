#!/usr/bin/env python3
"""
Standalone PrismAudio feature extraction script.
Runs in a separate Python env with JAX/TF installed (auto-created by PrismAudioFeatureExtractor).

Usage:
    python extract_features.py --video input.mp4 --cot_text "description..." --output features.npz
"""

import argparse
import os
import sys
import time
import numpy as np
import torch

# Add plugin root to sys.path so data_utils (and prismaudio_core) are importable
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PLUGIN_DIR = os.path.dirname(_SCRIPT_DIR)
if _PLUGIN_DIR not in sys.path:
    sys.path.insert(0, _PLUGIN_DIR)


def _step(n, total, label):
    """Print step header and return start time."""
    print(f"[extract] Step {n}/{total} — {label}...", flush=True)
    return time.perf_counter()


def _done(t0, extra=""):
    elapsed = time.perf_counter() - t0
    suffix = f"  {extra}" if extra else ""
    print(f"[extract]   done in {elapsed:.1f}s{suffix}", flush=True)


def main():
    t_total = time.perf_counter()

    parser = argparse.ArgumentParser(description="PrismAudio feature extraction")
    parser.add_argument("--video", required=True, help="Path to input video")
    parser.add_argument("--cot_text", required=True, help="Chain-of-thought description")
    parser.add_argument("--output", required=True, help="Output .npz path")
    parser.add_argument("--synchformer_ckpt", default=None, help="Path to synchformer checkpoint")
    parser.add_argument("--vae_config", default=None, help="Path to VAE config JSON")
    parser.add_argument("--source_fps", type=float, default=30.0, help="Original video fps (used when --video is a .npy file)")
    parser.add_argument("--clip_fps", type=float, default=4.0)
    parser.add_argument("--clip_size", type=int, default=288)
    parser.add_argument("--sync_fps", type=float, default=25.0)
    parser.add_argument("--sync_size", type=int, default=224)
    args = parser.parse_args()

    print(f"[extract] Python     : {sys.executable}", flush=True)
    print(f"[extract] Video      : {args.video}", flush=True)
    print(f"[extract] Output     : {args.output}", flush=True)
    print(f"[extract] CoT text   : {args.cot_text[:80]}{'...' if len(args.cot_text) > 80 else ''}", flush=True)

    if not os.path.exists(args.video):
        print(f"[extract] ERROR: video not found: {args.video}", flush=True)
        sys.exit(1)

    print(f"[extract] Device     : {'cuda' if torch.cuda.is_available() else 'cpu'}", flush=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ------------------------------------------------------------------
    t0 = _step(1, 6, "importing dependencies")
    from data_utils.v2a_utils.feature_utils_288 import FeaturesUtils
    import torchvision.transforms as T
    _done(t0)

    # ------------------------------------------------------------------
    t0 = _step(2, 6, "loading models (T5, VideoPrism, Synchformer)")
    feat_utils = FeaturesUtils(
        vae_config_path=args.vae_config,
        synchformer_ckpt=args.synchformer_ckpt,
        device=device,
    )
    _done(t0)

    # ------------------------------------------------------------------
    t0 = _step(3, 6, "reading and preprocessing video")
    if args.video.endswith(".npy"):
        all_frames = np.load(args.video)   # [T, H, W, C] uint8
        fps = args.source_fps
        total_frames = all_frames.shape[0]
        duration = total_frames / fps
        print(f"[extract]   fps={fps:.3f}  frames={total_frames}  duration={duration:.2f}s", flush=True)

        clip_indices = [int(i * fps / args.clip_fps) for i in range(int(duration * args.clip_fps))]
        clip_indices = [min(i, total_frames - 1) for i in clip_indices]
        clip_frames = all_frames[clip_indices]
        print(f"[extract]   CLIP frames  : {len(clip_indices)} @ {args.clip_fps}fps → {args.clip_size}×{args.clip_size}", flush=True)

        sync_indices = [int(i * fps / args.sync_fps) for i in range(int(duration * args.sync_fps))]
        sync_indices = [min(i, total_frames - 1) for i in sync_indices]
        sync_frames = all_frames[sync_indices]
        print(f"[extract]   Sync frames  : {len(sync_indices)} @ {args.sync_fps}fps → {args.sync_size}×{args.sync_size}", flush=True)
    else:
        from decord import VideoReader, cpu
        vr = VideoReader(args.video, ctx=cpu(0))
        fps = vr.get_avg_fps()
        total_frames = len(vr)
        duration = total_frames / fps
        print(f"[extract]   fps={fps:.3f}  frames={total_frames}  duration={duration:.2f}s", flush=True)

        clip_indices = [int(i * fps / args.clip_fps) for i in range(int(duration * args.clip_fps))]
        clip_indices = [min(i, total_frames - 1) for i in clip_indices]
        clip_frames = vr.get_batch(clip_indices).asnumpy()
        print(f"[extract]   CLIP frames  : {len(clip_indices)} @ {args.clip_fps}fps → {args.clip_size}×{args.clip_size}", flush=True)

        sync_indices = [int(i * fps / args.sync_fps) for i in range(int(duration * args.sync_fps))]
        sync_indices = [min(i, total_frames - 1) for i in sync_indices]
        sync_frames = vr.get_batch(sync_indices).asnumpy()
        print(f"[extract]   Sync frames  : {len(sync_indices)} @ {args.sync_fps}fps → {args.sync_size}×{args.sync_size}", flush=True)

    clip_transform = T.Compose([
        T.ToPILImage(),
        T.Resize(args.clip_size),
        T.CenterCrop(args.clip_size),
        T.ToTensor(),
        T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    clip_input = torch.stack([clip_transform(f) for f in clip_frames]).unsqueeze(0).to(device)

    sync_transform = T.Compose([
        T.ToPILImage(),
        T.Resize(args.sync_size),
        T.CenterCrop(args.sync_size),
        T.ToTensor(),
        T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    sync_input = torch.stack([sync_transform(f) for f in sync_frames]).unsqueeze(0).to(device)
    _done(t0)

    # ------------------------------------------------------------------
    t0 = _step(4, 6, "encoding text with T5-Gemma")
    text_features = feat_utils.encode_t5_text([args.cot_text])
    _done(t0, f"shape={tuple(text_features.shape)}")

    # ------------------------------------------------------------------
    t0 = _step(5, 6, "encoding video with VideoPrism")
    global_video_features, video_features, global_text_features = \
        feat_utils.encode_video_and_text_with_videoprism(clip_input, [args.cot_text])
    _done(t0, f"video={tuple(video_features.shape)}  global={tuple(global_video_features.shape)}")

    # ------------------------------------------------------------------
    t0 = _step(6, 6, "encoding video with Synchformer")
    sync_features = feat_utils.encode_video_with_sync(sync_input)
    _done(t0, f"shape={tuple(sync_features.shape)}")

    # ------------------------------------------------------------------
    t0 = time.perf_counter()
    print(f"[extract] Saving features to {args.output} ...", flush=True)
    np.savez(
        args.output,
        video_features=video_features.cpu().float().numpy(),
        global_video_features=global_video_features.cpu().float().numpy(),
        text_features=text_features.cpu().float().numpy(),
        global_text_features=global_text_features.cpu().float().numpy(),
        sync_features=sync_features.cpu().float().numpy(),
        caption_cot=args.cot_text,
        duration=duration,
    )
    print(f"[extract] Saved in {time.perf_counter() - t0:.1f}s", flush=True)
    print(f"[extract] Total time: {time.perf_counter() - t_total:.1f}s", flush=True)


if __name__ == "__main__":
    main()
