"""Fusion model combining per-modality encoders.

Provides a simple concat+MLP baseline and an optional cross-attention block.
Comments explain early vs late fusion trade-offs.
"""
from typing import Dict, Any, Optional
import torch
import torch.nn as nn


class CrossAttentionBlock(nn.Module):
    """Minimal cross-attention: queries from one modality attend to keys/vals of another.

    This is intentionally simple — for production use multi-head attention from
    torch.nn or transformers with positional encodings.
    """

    def __init__(self, dim: int):
        super().__init__()
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.out = nn.Linear(dim, dim)

    def forward(self, x_q: torch.Tensor, x_kv: torch.Tensor) -> torch.Tensor:
        # x_q: (B, D), x_kv: (B, D)
        q = self.q(x_q).unsqueeze(1)  # (B,1,D)
        k = self.k(x_kv).unsqueeze(1)
        v = self.v(x_kv).unsqueeze(1)
        attn = torch.softmax((q @ k.transpose(-2, -1)) / (q.shape[-1] ** 0.5), dim=-1)
        out = (attn @ v).squeeze(1)
        return self.out(out)


class FusionModel(nn.Module):
    """Fusion model supporting concatenation and optional cross-attention.

    Early fusion: combine raw inputs before heavy encoding (can allow models to
    learn joint low-level features). Late fusion: combine modality embeddings
    after separate encoders (more modular, safer if modalities are missing).
    """

    def __init__(self, text_dim: int = 64, audio_dim: int = 64, video_dim: int = 64,
                 hidden: int = 128, use_cross_attention: bool = False):
        super().__init__()
        self.use_cross = use_cross_attention
        self.proj_text = nn.Linear(text_dim, hidden)
        self.proj_audio = nn.Linear(audio_dim, hidden)
        self.proj_video = nn.Linear(video_dim, hidden)
        if use_cross_attention:
            self.cross = CrossAttentionBlock(hidden)
        self.mlp = nn.Sequential(nn.Linear(hidden * 3, hidden), nn.ReLU())
        self.reg_head = nn.Linear(hidden, 5)
        self.cls_head = nn.Linear(hidden, 3)

    def forward(self, text_emb: torch.Tensor, audio_emb: torch.Tensor, video_emb: torch.Tensor) -> Dict[str, torch.Tensor]:
        # Project
        t = torch.relu(self.proj_text(text_emb))
        a = torch.relu(self.proj_audio(audio_emb))
        v = torch.relu(self.proj_video(video_emb))
        if self.use_cross:
            # Cross-attention between text and audio for demo (symmetric)
            t = t + self.cross(t, a)
            a = a + self.cross(a, v)
        cat = torch.cat([t, a, v], dim=-1)
        h = self.mlp(cat)
        return {"traits": self.reg_head(h), "logits": self.cls_head(h)}
