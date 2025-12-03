"""Audio dataset: loads waveform, computes features, and returns per-utterance tensors.

For personality labels (speaker-level) we may aggregate several utterances per
speaker. For the demo we treat each file as one sample.
"""
from typing import Optional, Dict, Any
import os
import torch
from torch.utils.data import Dataset
import numpy as np
from ..utils.io_utils import read_audio
from ..utils.audio_utils import compute_log_mel
from ..config import get_default_config, Config


class AudioDataset(Dataset):
    """Loads audio files and returns feature tensors and labels.

    Inputs:
    - path: file path to a WAV file or directory containing WAVs (demo uses one file)
    - config: config object controlling sample rate and mel bins
    """

    def __init__(self, path: str, config: Optional[Config] = None):
        self.config = config or get_default_config()
        if os.path.isdir(path):
            files = [os.path.join(path, f) for f in os.listdir(path) if f.endswith(".wav")]
            self.files = sorted(files)
        else:
            self.files = [path]

        # For demo: simple dummy labels aligned with text dataset (index parity)
        self.labels = [0 if i % 2 == 0 else 1 for i in range(len(self.files))]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        path = self.files[idx]
        waveform, sr = read_audio(path, target_sr=self.config.audio_sr)
        # compute features
        mel = compute_log_mel(waveform, sr=sr, n_mels=self.config.audio_n_mels)
        # convert to torch tensor (C, T)
        mel_t = torch.tensor(mel)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return {"features": mel_t, "label": label, "path": path}
