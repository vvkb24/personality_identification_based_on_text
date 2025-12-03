"""Simple MLP encoder for tabular/questionnaire features.

Produces a joint embedding and heads for trait regression and emotion
classification (if requested).
"""
from typing import Optional
import torch
import torch.nn as nn


class TabularModel(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: Optional[list] = None, trait_out: int = 5, n_emotions: int = 3):
        super().__init__()
        hidden_dims = hidden_dims or [256, 128]
        layers = []
        in_dim = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm1d(h))
            in_dim = h
        self.encoder = nn.Sequential(*layers)
        self.trait_head = nn.Linear(in_dim, trait_out)
        self.cls_head = nn.Linear(in_dim, n_emotions)

    def forward(self, x):
        h = self.encoder(x)
        traits = self.trait_head(h)
        logits = self.cls_head(h)
        return {"traits": traits, "logits": logits}
