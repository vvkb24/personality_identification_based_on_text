"""Train script for text-only model.

This script is intentionally small: it builds a dataset from the synthetic
JSONL, a dataloader, a small model, and runs one epoch. It demonstrates the
typical training loop with optimizer, scheduler, and checkpointing.

Run: `python src/train/train_text.py`
"""
import os
import torch
from torch.utils.data import DataLoader
from ..config import get_default_config
from ..data.text_dataset import TextDataset
from ..models.text_model import TextModel, loss_fn
from ..utils.io_utils import save_checkpoint


def train_one_epoch(model, dl, optim, device):
    model.train()
    total_loss = 0.0
    for batch in dl:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        traits = batch["traits"].to(device)
        emotion = batch["emotion"].to(device)
        optim.zero_grad()
        out = model(input_ids, attention_mask)
        loss, parts = loss_fn(out, traits, emotion)
        loss.backward()
        optim.step()
        total_loss += loss.item()
    return total_loss / len(dl)


def main():
    cfg = get_default_config()
    os.makedirs(cfg.output_dir, exist_ok=True)
    ds = TextDataset(os.path.join(cfg.data_dir, "text_samples.jsonl"), cfg)
    dl = DataLoader(ds, batch_size=cfg.batch_size, shuffle=True)
    model = TextModel().to(cfg.device)
    optim = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    loss = train_one_epoch(model, dl, optim, cfg.device)
    print(f"Finished epoch, avg loss={loss:.4f}")
    save_checkpoint(model, os.path.join(cfg.output_dir, "text_model.pt"), optimizer=optim)


if __name__ == "__main__":
    main()
