"""Video dataset: extracts frame-level arrays and returns temporal tensors.

For speed the demo extracts a small number of frames and optionally computes a
per-frame embedding using a tiny CNN in the model.
"""
from typing import Optional, Dict, Any, List
import os
import torch
from torch.utils.data import Dataset
from ..utils.video_utils import extract_frames
from ..config import get_default_config, Config
import numpy as np


class VideoDataset(Dataset):
    """Loads a video, extracts frames, and returns them as tensors.

    Each item is a small sequence of frames (T, C, H, W) as floats in [0,1].
    """

    def __init__(self, path: str, config: Optional[Config] = None, max_frames: int = 16):
        self.config = config or get_default_config()
        if os.path.isdir(path):
            files = [os.path.join(path, f) for f in os.listdir(path) if f.endswith(".mp4")]
            self.files = sorted(files)
        else:
            self.files = [path]
        self.max_frames = max_frames

        # demo labels
        self.labels = [0 for _ in self.files]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        path = self.files[idx]
        frames = extract_frames(path, max_frames=self.max_frames)
        # Convert list of HxWxC uint8 to tensor T x C x H x W
        arrs = [f.astype(np.float32) / 255.0 for f in frames]
        if len(arrs) == 0:
            # fallback: small black frame
            arrs = [np.zeros((self.config.audio_n_mels, self.config.audio_n_mels, 3), dtype=np.float32)]
        tensor = torch.tensor(np.stack(arrs)).permute(0, 3, 1, 2)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return {"frames": tensor, "label": label, "path": path}
