"""SelVA Audio Post-Processing nodes.

Post-generation enhancement applied to standard AUDIO outputs:
  SelvaHarmonicExciter    — multi-band harmonic exciter (HPF → tanh → mix)
  SelvaOutputNormalizer   — LUFS normalization + true peak limiting
"""

import numpy as np
import torch

from .utils import SELVA_CATEGORY


class SelvaHarmonicExciter:
    """Multi-band harmonic exciter for post-generation enhancement.

    Isolates high-frequency content above a cutoff, applies tanh saturation
    to generate 2nd/3rd harmonics, then mixes back with the dry signal.
    Restores harmonic richness lost during BigVGAN vocoder reconstruction.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                "cutoff_hz": ("FLOAT", {
                    "default": 3000.0, "min": 500.0, "max": 16000.0, "step": 100.0,
                    "tooltip": "Highpass cutoff frequency in Hz. Only content above this is excited. "
                               "3000 Hz targets the upper harmonics BigVGAN tends to smear.",
                }),
                "drive": ("FLOAT", {
                    "default": 2.0, "min": 1.0, "max": 10.0, "step": 0.5,
                    "tooltip": "Saturation drive. Higher = more harmonics generated. "
                               "2-3 is subtle, 5+ is aggressive.",
                }),
                "mix": ("FLOAT", {
                    "default": 0.15, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": "Wet/dry blend. 0.1-0.2 is subtle enhancement, "
                               "0.5+ is aggressive harmonic addition.",
                }),
            }
        }

    RETURN_TYPES  = ("AUDIO",)
    RETURN_NAMES  = ("audio",)
    FUNCTION      = "excite"
    CATEGORY      = SELVA_CATEGORY
    DESCRIPTION   = (
        "Multi-band harmonic exciter. Applies tanh saturation to the high-frequency band "
        "to restore harmonics lost during BigVGAN vocoder reconstruction. "
        "Uses pedalboard.HighpassFilter for band isolation."
    )

    def excite(self, audio, cutoff_hz: float, drive: float, mix: float):
        from pedalboard import Pedalboard, HighpassFilter

        wav = audio["waveform"][0]   # [C, T]
        sr  = audio["sample_rate"]

        wav_np = wav.float().numpy()   # [C, T]

        # Isolate HF band
        board = Pedalboard([HighpassFilter(cutoff_frequency_hz=cutoff_hz)])
        hf = board(wav_np, sr)         # [C, T]

        # Tanh saturation — normalize by drive so output stays in [-1, 1]
        excited = np.tanh(hf * drive) / max(drive, 1.0)

        # Mix back with dry
        mixed = wav_np + mix * excited

        # Soft clip to prevent going over
        mixed = np.tanh(mixed)

        wav_out = torch.from_numpy(mixed).unsqueeze(0)  # [1, C, T]
        print(
            f"[HarmonicExciter] cutoff={cutoff_hz}Hz  drive={drive}  mix={mix:.0%}",
            flush=True,
        )
        return ({"waveform": wav_out, "sample_rate": sr},)



class SelvaOutputNormalizer:
    """Normalize generated audio to a target LUFS level with true peak limiting.

    Apply as the final node before saving — brings generated audio to a
    consistent loudness target regardless of input video loudness variation.
    Uses pyloudnorm (BS.1770-4).
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                "target_lufs": ("FLOAT", {
                    "default": -14.0, "min": -40.0, "max": -6.0, "step": 0.5,
                    "tooltip": "Target integrated loudness in LUFS. "
                               "-14 LUFS for streaming (Spotify/YouTube), "
                               "-9 to -7 for production masters.",
                }),
                "true_peak_dbtp": ("FLOAT", {
                    "default": -1.0, "min": -6.0, "max": 0.0, "step": 0.5,
                    "tooltip": "True peak ceiling in dBTP applied after LUFS gain.",
                }),
            }
        }

    RETURN_TYPES  = ("AUDIO",)
    RETURN_NAMES  = ("audio",)
    FUNCTION      = "normalize"
    CATEGORY      = SELVA_CATEGORY
    DESCRIPTION   = (
        "Normalize output audio to a target LUFS level (BS.1770-4) with true peak limiting. "
        "Apply as the last node before saving. Uses pyloudnorm."
    )

    def normalize(self, audio, target_lufs: float, true_peak_dbtp: float):
        import pyloudnorm as pyln

        wav = audio["waveform"][0]   # [C, T]
        sr  = audio["sample_rate"]

        tp_linear = 10.0 ** (true_peak_dbtp / 20.0)

        wav_np = wav.permute(1, 0).double().numpy()   # [T, C]
        if wav_np.shape[1] == 1:
            wav_np = wav_np[:, 0]                     # [T] mono

        meter    = pyln.Meter(sr)
        loudness = meter.integrated_loudness(wav_np)

        if not np.isfinite(loudness):
            print("[OutputNormalizer] Could not measure loudness — clip too short or silent. Passing through.", flush=True)
            return (audio,)

        gain_db     = target_lufs - loudness
        gain_linear = 10.0 ** (gain_db / 20.0)

        wav_out = wav * gain_linear

        peak = wav_out.abs().max().item()
        if peak > tp_linear:
            wav_out = wav_out * (tp_linear / peak)

        print(
            f"[OutputNormalizer] {loudness:.1f} LUFS → {target_lufs} LUFS  "
            f"gain={gain_db:+.1f}dB  TP={true_peak_dbtp}dBTP",
            flush=True,
        )
        return ({"waveform": wav_out.unsqueeze(0), "sample_rate": sr},)
