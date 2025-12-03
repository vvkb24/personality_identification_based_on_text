"""Audio feature utilities: load audio, compute log-mel and MFCCs.

These functions use librosa when available and fall back to simple
implementations where possible. Comments explain why features like MFCCs and
log-mel spectrograms help with emotion/personality inference.
"""
from typing import Tuple
import numpy as np


def compute_log_mel(waveform: np.ndarray, sr: int = 16000, n_mels: int = 64,
                    n_fft: int = 1024, hop_length: int = 256) -> np.ndarray:
    """Compute log-mel spectrogram.

    Returns a (n_mels, T) array. Log-mel compresses dynamic range and aligns
    frequency bins to human perception; useful for emotion because prosodic and
    spectral cues often appear in mel bands.
    """
    try:
        import librosa

        mel = librosa.feature.melspectrogram(y=waveform, sr=sr, n_fft=n_fft,
                                             hop_length=hop_length, n_mels=n_mels)
        log_mel = np.log1p(mel)
        return log_mel.astype(np.float32)
    except Exception:
        # Minimal fallback: short-time energy across frames as crude proxy
        frames = []
        for i in range(0, len(waveform) - hop_length, hop_length):
            frame = waveform[i:i + n_fft]
            frames.append(np.sum(frame ** 2))
        feat = np.array(frames)[None, :]
        # Expand to n_mels via simple tiling (very crude fallback)
        return np.tile(feat, (n_mels, 1)).astype(np.float32)


def compute_mfcc(waveform: np.ndarray, sr: int = 16000, n_mfcc: int = 13) -> np.ndarray:
    """Compute MFCCs. MFCCs capture the spectral envelope and are widely used
    in speech tasks and emotion recognition as they summarize timbral aspects.
    """
    try:
        import librosa

        mfcc = librosa.feature.mfcc(y=waveform, sr=sr, n_mfcc=n_mfcc)
        return mfcc.astype(np.float32)
    except Exception:
        # Simple fallback: downsample and return a few aggregated features
        small = waveform[:: max(1, len(waveform) // 100)]
        mean = np.mean(small)
        std = np.std(small)
        return np.tile(np.array([[mean], [std]]), (n_mfcc, 1)).astype(np.float32)
