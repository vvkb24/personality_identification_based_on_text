"""Audio model: small CNN + pooling or wrapper around wav2vec2-stub.

This lightweight model demonstrates audio embedding extraction and downstream
heads. Replace with wav2vec2 or other pretrained model for better performance.
"""
from typing import Dict, Any
import torch
import torch.nn as nn


class AudioModel(nn.Module):
    """Simple CNN over mel spectrogram inputs.

    Input shape: (B, n_mels, T). We treat n_mels as channels for the conv.
    """

    def __init__(self, n_mels: int = 64, hidden: int = 64, n_emotions: int = 3):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(n_mels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(16),
            nn.Flatten(),
        )
        self.fc = nn.Sequential(nn.Linear(32 * 16, hidden), nn.ReLU())
        self.reg_head = nn.Linear(hidden, 5)
        self.cls_head = nn.Linear(hidden, n_emotions)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        # x: (B, n_mels, T) ; ensure shape
        if x.ndim == 2:
            x = x.unsqueeze(0)
        if x.shape[1] != 64:
            # pad or truncate mel bands to 64 for demo
            pad = 64 - x.shape[1]
            if pad > 0:
                x = torch.nn.functional.pad(x, (0, 0, 0, pad))
            else:
                x = x[:, :64, :]
        h = self.conv(x)
        emb = self.fc(h)
        return {"traits": self.reg_head(emb), "logits": self.cls_head(emb)}
