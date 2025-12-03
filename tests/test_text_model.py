"""Unit tests for text model using synthetic data."""
import torch
from src.models.text_model import TextModel


def test_text_forward():
    model = TextModel()
    input_ids = torch.randint(0, 1000, (2, 16))
    out = model(input_ids)
    assert "traits" in out and "logits" in out
    assert out["traits"].shape[0] == 2
