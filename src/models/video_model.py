"""Video model: per-frame CNN + temporal pooling.

This lightweight approach processes frames independently with a small CNN and
aggregates temporally. 3D convs are more powerful but computationally costly.
"""
from typing import Dict, Any
import torch
import torch.nn as nn


class FrameCNN(nn.Module):
    """Tiny per-frame CNN to extract embeddings from an RGB frame."""

    def __init__(self, out_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((8, 8)),
            nn.Flatten(),
            nn.Linear(16 * 8 * 8, out_dim),
            nn.ReLU(),
        )

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        # frames: (B, T, C, H, W) -> process per-frame via reshape
        B, T, C, H, W = frames.shape
        x = frames.view(B * T, C, H, W)
        emb = self.net(x)  # (B*T, D)
        emb = emb.view(B, T, -1)
        return emb


class VideoModel(nn.Module):
    """Per-frame encoder + temporal pooling + heads."""

    def __init__(self, emb_dim: int = 64, hidden: int = 64, n_emotions: int = 3):
        super().__init__()
        self.frame_cnn = FrameCNN(out_dim=emb_dim)
        self.project = nn.Linear(emb_dim, hidden)
        self.reg_head = nn.Linear(hidden, 5)
        self.cls_head = nn.Linear(hidden, n_emotions)

    def forward(self, frames: torch.Tensor) -> Dict[str, torch.Tensor]:
        emb_seq = self.frame_cnn(frames)  # (B, T, D)
        pooled = emb_seq.mean(dim=1)
        h = torch.relu(self.project(pooled))
        return {"traits": self.reg_head(h), "logits": self.cls_head(h)}
