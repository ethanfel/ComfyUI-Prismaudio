"""SelVA Audio Preprocessors — condition training clips for codec compatibility.

Two nodes that reduce the domain mismatch between custom training audio and the
MMAudio VAE's expected spectral distribution, improving LoRA training quality:

  SelvaHfSmoother      — soft low-pass blend to attenuate extreme HF content
  SelvaSpectralMatcher — adaptive per-band EQ toward the codec's training distribution

Root cause they address: MMAudio was trained on natural sounds (speech, foley, env)
with limited engineered HF content. The BigVGANv2 vocoder (frozen, pre-trained) handles
the codec's HF reconstruction poorly for sound design / music training clips, because
those clips land in a latent-space region the vocoder never saw during training.

Recommended order: SpectralMatcher → HfSmoother → feature extraction → LoRA training.
"""

import numpy as np
import torch
import torchaudio.functional as AF

from .utils import SELVA_CATEGORY


# ── Mel filterbank (same algorithm as selva_core/ext/mel_converter.py) ────────

def _mel_filterbank(sr: int, n_fft: int, n_mels: int,
                    fmin: float, fmax: float) -> torch.Tensor:
    """Returns mel filterbank matrix [n_mels, n_fft//2+1]."""
    def hz_to_mel(f):
        return 2595.0 * np.log10(1.0 + np.asarray(f) / 700.0)

    def mel_to_hz(m):
        return 700.0 * (10.0 ** (np.asarray(m) / 2595.0) - 1.0)

    n_freqs   = n_fft // 2 + 1
    fft_freqs = np.linspace(0.0, sr / 2.0, n_freqs)
    mel_pts   = np.linspace(hz_to_mel(fmin), hz_to_mel(fmax), n_mels + 2)
    hz_pts    = mel_to_hz(mel_pts)

    fb = np.zeros((n_mels, n_freqs), dtype=np.float32)
    for m in range(1, n_mels + 1):
        lo, mid, hi = hz_pts[m - 1], hz_pts[m], hz_pts[m + 1]
        up   = (fft_freqs - lo)  / (mid - lo  + 1e-12)
        down = (hi - fft_freqs)  / (hi  - mid + 1e-12)
        fb[m - 1] = np.maximum(0.0, np.minimum(up, down))
    return torch.from_numpy(fb)


# ── VAE target log-mel means (source: selva_core/ext/autoencoder/vae.py) ──────
# These are the per-band expected log-mel energy means from MMAudio's training data.
# Used as the spectral matching target: clips are EQ'd to match this profile.

_TARGET_MEAN_80D = [
    -1.6058, -1.3676, -1.2520, -1.2453, -1.2078, -1.2224, -1.2419, -1.2439,
    -1.2922, -1.2927, -1.3170, -1.3543, -1.3401, -1.3836, -1.3907, -1.3912,
    -1.4313, -1.4152, -1.4527, -1.4728, -1.4568, -1.5101, -1.5051, -1.5172,
    -1.5623, -1.5373, -1.5746, -1.5687, -1.6032, -1.6131, -1.6081, -1.6331,
    -1.6489, -1.6489, -1.6700, -1.6738, -1.6953, -1.6969, -1.7048, -1.7280,
    -1.7361, -1.7495, -1.7658, -1.7814, -1.7889, -1.8064, -1.8221, -1.8377,
    -1.8417, -1.8643, -1.8857, -1.8929, -1.9173, -1.9379, -1.9531, -1.9673,
    -1.9824, -2.0042, -2.0215, -2.0436, -2.0766, -2.1064, -2.1418, -2.1855,
    -2.2319, -2.2767, -2.3161, -2.3572, -2.3954, -2.4282, -2.4659, -2.5072,
    -2.5552, -2.6074, -2.6584, -2.7107, -2.7634, -2.8266, -2.8981, -2.9673,
]

_TARGET_MEAN_128D = [
    -3.3462, -2.6723, -2.4893, -2.3143, -2.2664, -2.3317, -2.1802, -2.4006,
    -2.2357, -2.4597, -2.3717, -2.4690, -2.5142, -2.4919, -2.6610, -2.5047,
    -2.7483, -2.5926, -2.7462, -2.7033, -2.7386, -2.8112, -2.7502, -2.9594,
    -2.7473, -3.0035, -2.8891, -2.9922, -2.9856, -3.0157, -3.1191, -2.9893,
    -3.1718, -3.0745, -3.1879, -3.2310, -3.1424, -3.2296, -3.2791, -3.2782,
    -3.2756, -3.3134, -3.3509, -3.3750, -3.3951, -3.3698, -3.4505, -3.4509,
    -3.5089, -3.4647, -3.5536, -3.5788, -3.5867, -3.6036, -3.6400, -3.6747,
    -3.7072, -3.7279, -3.7283, -3.7795, -3.8259, -3.8447, -3.8663, -3.9182,
    -3.9605, -3.9861, -4.0105, -4.0373, -4.0762, -4.1121, -4.1488, -4.1874,
    -4.2461, -4.3170, -4.3639, -4.4452, -4.5282, -4.6297, -4.7019, -4.7960,
    -4.8700, -4.9507, -5.0303, -5.0866, -5.1634, -5.2342, -5.3242, -5.4053,
    -5.4927, -5.5712, -5.6464, -5.7052, -5.7619, -5.8410, -5.9188, -6.0103,
    -6.0955, -6.1673, -6.2362, -6.3120, -6.3926, -6.4797, -6.5565, -6.6511,
    -6.8130, -6.9961, -7.1275, -7.2457, -7.3576, -7.4663, -7.6136, -7.7469,
    -7.8815, -8.0132, -8.1515, -8.3071, -8.4722, -8.7418, -9.3975, -9.6628,
    -9.7671, -9.8863, -9.9992, -10.0860, -10.1709, -10.5418, -11.2795, -11.3861,
]

_MEL_CONFIGS = {
    "16k": dict(sr=16_000, n_fft=1024, n_mels=80,  hop=256, fmin=0, fmax=8_000,
                target=_TARGET_MEAN_80D,  log10=True),
    "44k": dict(sr=44_100, n_fft=2048, n_mels=128, hop=512, fmin=0, fmax=22_050,
                target=_TARGET_MEAN_128D, log10=False),
}


# ── Node 1: HF Smoother ───────────────────────────────────────────────────────

class SelvaHfSmoother:
    """Soft high-frequency attenuation for LoRA training clip preprocessing.

    Blends a low-pass filtered copy of the audio with the original.  Attenuates
    the extreme HF content common in engineered sound design that the BigVGANv2
    vocoder handles poorly, bringing the clip closer to the spectral region the
    MMAudio codec was trained on (natural sounds with limited HF energy).

    A blend of 0.7 at 12 kHz is a transparent starting point — audible only on
    close comparison. Increase blend or lower cutoff if roundtrip quality is still
    poor after spectral matching.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                "cutoff_hz": ("FLOAT", {
                    "default": 12000.0, "min": 2000.0, "max": 20000.0, "step": 500.0,
                    "tooltip": "Low-pass cutoff. 12 kHz is gentle; lower = more aggressive.",
                }),
                "blend": ("FLOAT", {
                    "default": 0.7, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": "0 = original, 1 = fully filtered. 0.7 is a transparent starting point.",
                }),
            }
        }

    RETURN_TYPES  = ("AUDIO",)
    RETURN_NAMES  = ("audio",)
    FUNCTION      = "process"
    CATEGORY      = SELVA_CATEGORY
    DESCRIPTION   = (
        "Blends a low-pass filtered version of the audio with the original to gently attenuate "
        "high-frequency content that the SelVA codec handles poorly. "
        "Use before feature extraction to improve LoRA training targets. "
        "Run after SelVA Spectral Matcher for best results."
    )

    def process(self, audio, cutoff_hz: float, blend: float):
        waveform = audio["waveform"].float()   # [1, C, L]
        sr       = audio["sample_rate"]

        filtered = AF.lowpass_biquad(waveform, sr, cutoff_hz)
        out = blend * filtered + (1.0 - blend) * waveform

        # Preserve RMS level — LPF removes energy, keep the clip at its original loudness
        rms_in  = waveform.pow(2).mean().sqrt().clamp(min=1e-8)
        rms_out = out.pow(2).mean().sqrt().clamp(min=1e-8)
        out     = out * (rms_in / rms_out)

        peak = out.abs().max()
        if peak > 1.0:
            out = out / peak

        print(f"[HF Smoother] cutoff={cutoff_hz:.0f} Hz  blend={blend:.2f}  "
              f"rms={rms_in:.4f}→{out.pow(2).mean().sqrt():.4f}  "
              f"peak={out.abs().max():.4f}", flush=True)

        return ({"waveform": out, "sample_rate": sr},)


# ── Node 2: Spectral Matcher ──────────────────────────────────────────────────

class SelvaSpectralMatcher:
    """Adaptive per-band EQ toward the SelVA VAE's expected spectral distribution.

    Computes the log-mel energy profile of the clip and compares it to the per-band
    means stored in the VAE's normalization buffers (the statistics MMAudio was trained
    on).  Applies a smooth frequency-domain gain correction so the clip's spectral shape
    matches what the codec expects, improving encode→decode roundtrip quality and
    therefore LoRA training target quality.

    The correction is additive in log space (multiplicative in linear), so it only
    changes spectral balance — not the waveform's timing or phase structure.

    max_gain_db clamps the correction to prevent extreme boosts on very quiet bands.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                "mode": (["44k", "16k"], {
                    "tooltip": "Must match the SelVA model you are training. "
                               "44k = large model, 16k = small model.",
                }),
                "strength": ("FLOAT", {
                    "default": 0.8, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": "0 = no correction, 1 = full match to VAE distribution. "
                               "0.8 is a good starting point.",
                }),
                "max_gain_db": ("FLOAT", {
                    "default": 12.0, "min": 1.0, "max": 30.0, "step": 1.0,
                    "tooltip": "Clamps per-band gain to ±dB. Prevents extreme boosts on "
                               "very quiet frequency bands. 12 dB is conservative.",
                }),
            }
        }

    RETURN_TYPES  = ("AUDIO",)
    RETURN_NAMES  = ("audio",)
    FUNCTION      = "process"
    CATEGORY      = SELVA_CATEGORY
    DESCRIPTION   = (
        "Applies a smooth per-band gain correction to bring the audio's spectral profile "
        "in line with the MMAudio VAE's expected distribution, derived from the per-band "
        "normalization statistics baked into the VAE weights. "
        "Use before feature extraction to improve LoRA training target quality. "
        "Run before SelVA HF Smoother."
    )

    def process(self, audio, mode: str, strength: float, max_gain_db: float):
        cfg      = _MEL_CONFIGS[mode]
        waveform = audio["waveform"].float()   # [1, C, L]
        sr_in    = audio["sample_rate"]
        sr_tgt   = cfg["sr"]
        n_fft    = cfg["n_fft"]
        hop      = cfg["hop"]

        # ── flatten to mono and resample if needed ────────────────────────────
        wav = waveform[0].mean(0)   # [L]
        if sr_in != sr_tgt:
            wav = AF.resample(wav.unsqueeze(0), sr_in, sr_tgt).squeeze(0)

        device = wav.device
        window = torch.hann_window(n_fft, device=device)

        # ── STFT ──────────────────────────────────────────────────────────────
        stft = torch.stft(wav, n_fft, hop_length=hop, win_length=n_fft,
                          window=window, center=True, return_complex=True)  # [n_freqs, T]
        mag  = stft.abs()                                                    # [n_freqs, T]

        # ── current log-mel mean per band ─────────────────────────────────────
        fb = _mel_filterbank(sr_tgt, n_fft, cfg["n_mels"],
                             cfg["fmin"], cfg["fmax"]).to(device)   # [n_mels, n_freqs]

        mel_mag = torch.matmul(fb, mag).clamp(min=1e-5)   # [n_mels, T]
        if cfg["log10"]:
            mel_log = torch.log10(mel_mag)
        else:
            mel_log = torch.log(mel_mag)

        current_mean = mel_log.mean(dim=-1)                               # [n_mels]
        target_mean  = torch.tensor(cfg["target"], device=device)         # [n_mels]

        # ── per-mel-band gain (log space) ─────────────────────────────────────
        mel_gain = (target_mean - current_mean) * strength                # [n_mels]

        # Clamp to ±max_gain_db
        if cfg["log10"]:
            max_log = max_gain_db / 20.0        # log10: 20 log10 = dB
        else:
            max_log = max_gain_db / 8.6859      # ln: 20 * log10(e) ≈ 8.686
        mel_gain = mel_gain.clamp(-max_log, max_log)

        # ── map mel gains → STFT frequency bins (weighted average) ────────────
        fb_sum     = fb.sum(0).clamp(min=1e-8)             # [n_freqs]
        freq_gain  = (mel_gain @ fb) / fb_sum              # [n_freqs]

        if cfg["log10"]:
            linear_gain = 10.0 ** freq_gain                # [n_freqs]
        else:
            linear_gain = torch.exp(freq_gain)             # [n_freqs]

        # ── apply gain in frequency domain and reconstruct ───────────────────
        stft_out = stft * linear_gain.unsqueeze(-1)        # [n_freqs, T]
        wav_out  = torch.istft(stft_out, n_fft, hop_length=hop, win_length=n_fft,
                               window=window, center=True,
                               length=wav.shape[0])         # [L]

        # ── resample back to original sr ──────────────────────────────────────
        if sr_in != sr_tgt:
            wav_out = AF.resample(wav_out.unsqueeze(0), sr_tgt, sr_in).squeeze(0)

        # ── preserve original RMS level ───────────────────────────────────────
        rms_in  = wav.pow(2).mean().sqrt().clamp(min=1e-8)
        rms_out = wav_out.pow(2).mean().sqrt().clamp(min=1e-8)
        wav_out = wav_out * (rms_in / rms_out)

        peak = wav_out.abs().max()
        if peak > 1.0:
            wav_out = wav_out / peak

        # ── reshape to match input layout [1, C, L] ───────────────────────────
        out = wav_out.unsqueeze(0).unsqueeze(0)
        if waveform.shape[1] > 1:
            out = out.expand(-1, waveform.shape[1], -1).clone()

        gain_db_range = (
            20.0 * torch.log10(linear_gain.clamp(min=1e-8))
        )
        print(f"[Spectral Matcher] mode={mode}  strength={strength:.2f}  "
              f"gain [{gain_db_range.min():.1f}, {gain_db_range.max():.1f}] dB  "
              f"rms={rms_in:.4f}→{out.pow(2).mean().sqrt():.4f}", flush=True)

        return ({"waveform": out, "sample_rate": sr_in},)
