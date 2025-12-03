"""Central configuration for the demo project.

This module defines a Config dataclass containing dataset paths, model
hyperparameters, training options, device selection, and reproducibility
settings. Each field includes comments about why it matters.
"""
from dataclasses import dataclass
from typing import Optional
import torch


@dataclass
class Config:
    """Configuration container for training and inference.

    Fields:
    - data_dir: Where example/synthetic data lives; replace with real dataset path.
    - output_dir: Where to save models and logs; keep separate from data for safety.
    - seed: Deterministic seed to improve reproducibility across runs.
    - device: Torch device; default selects GPU if available for speed.
    - batch_size: Batch size affects GPU memory and gradient noise.
    - lr: Learning rate for optimizers; important to tune for dataset/model scale.
    - epochs: Number of training epochs.
    - text_max_length: Tokenization length limit — longer increases memory/time.
    - audio_sr: Audio sample rate; choose standard values (16k or 16_000) for speech.
    - audio_n_mels: Feature dimension for mel-spectrograms.
    - nan_check: Whether to check tensors for NaNs during training (useful debugging).
    """

    data_dir: str = "data/example"
    output_dir: str = "models"
    seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size: int = 8
    lr: float = 1e-3
    epochs: int = 1
    text_max_length: int = 128
    audio_sr: int = 16000
    audio_n_mels: int = 64
    nan_check: bool = True


def get_default_config() -> Config:
    """Return a Config with defaults. Callers can override attributes.

    Using a function is convenient for programmatic changes and tests.
    """
    return Config()
