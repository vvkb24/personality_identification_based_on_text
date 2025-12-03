"""Small text model that can perform Big-Five regression and emotion classification.

The implementation uses a lightweight transformer stub when pretrained models
are unavailable. Losses: MSE for regression and CrossEntropy for classification.
"""
from typing import Optional, Dict, Any
import torch
import torch.nn as nn


class TextEncoderStub(nn.Module):
    """Very small encoder that maps token ids to an embedding and does mean pooling.

    This stub is a placeholder for a pretrained transformer; replace with a
    transformer (e.g., DistilBERT) for production.
    """

    def __init__(self, vocab_size: int = 30522, embed_dim: int = 128):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None):
        x = self.embed(input_ids)  # (B, L, D)
        x = x.permute(0, 2, 1)  # (B, D, L)
        x = self.pool(x).squeeze(-1)
        return x


class TextModel(nn.Module):
    """Combined model: encoder + projection + two heads.

    - Regression head predicts 5 continuous trait scores (MSE loss)
    - Classification head predicts emotion class (CrossEntropy)
    """

    def __init__(self, vocab_size: int = 30522, embed_dim: int = 128, hidden: int = 64, n_emotions: int = 3):
        super().__init__()
        self.encoder = TextEncoderStub(vocab_size=vocab_size, embed_dim=embed_dim)
        self.project = nn.Sequential(nn.Linear(embed_dim, hidden), nn.ReLU())
        self.reg_head = nn.Linear(hidden, 5)  # Big-Five
        self.cls_head = nn.Linear(hidden, n_emotions)

    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        emb = self.encoder(input_ids, attention_mask)
        h = self.project(emb)
        traits = self.reg_head(h)
        logits = self.cls_head(h)
        return {"traits": traits, "logits": logits}


def loss_fn(pred: Dict[str, torch.Tensor], target_traits: torch.Tensor, target_emotion: torch.Tensor):
    """Compute combined loss: MSE for traits + CE for emotion.

    The losses are summed; in practice you may weight them depending on task priority.
    """
    mse = nn.MSELoss()
    ce = nn.CrossEntropyLoss()
    loss_reg = mse(pred["traits"], target_traits)
    loss_cls = ce(pred["logits"], target_emotion)
    return loss_reg + loss_cls, {"mse": loss_reg.item(), "ce": loss_cls.item()}
