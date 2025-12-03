"""Train script for text-only model.

This script is intentionally small: it builds a dataset from the synthetic
JSONL, a dataloader, a small model, and runs one epoch. It demonstrates the
typical training loop with optimizer, scheduler, and checkpointing.

Run: `python src/train/train_text.py`
"""
import os
import torch
from torch.utils.data import DataLoader
import argparse
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-size", type=int, default=80, help="Number of training samples")
    parser.add_argument("--test-size", type=int, default=30, help="Number of test samples")
    parser.add_argument("--epochs", type=int, default=None, help="Number of epochs (overrides config)")
    args = parser.parse_args()

    cfg = get_default_config()
    if args.epochs is not None:
        cfg.epochs = args.epochs
    os.makedirs(cfg.output_dir, exist_ok=True)

    # Load full dataset and split into train/test using a fixed seed for reproducibility
    full = TextDataset(os.path.join(cfg.data_dir, "text_samples.jsonl"), cfg)
    total = len(full)
    train_n = args.train_size
    test_n = args.test_size
    if train_n + test_n > total:
        raise ValueError(f"Requested split {train_n}+{test_n} > available samples ({total})")

    # Deterministic shuffle
    import random

    rng = random.Random(cfg.seed)
    indices = list(range(total))
    rng.shuffle(indices)
    train_idx = indices[:train_n]
    test_idx = indices[train_n: train_n + test_n]

    # Create subset datasets
    from torch.utils.data import Subset

    train_ds = Subset(full, train_idx)
    test_ds = Subset(full, test_idx)

    train_dl = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)
    test_dl = DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=False)

    model = TextModel().to(cfg.device)
    optim = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    # Training loop
    for epoch in range(cfg.epochs):
        loss = train_one_epoch(model, train_dl, optim, cfg.device)
        print(f"Epoch {epoch+1}/{cfg.epochs} finished, train avg loss={loss:.4f}")

    # Save final checkpoint
    ckpt_path = os.path.join(cfg.output_dir, "text_model.pt")
    save_checkpoint(model, ckpt_path, optimizer=optim)
    print(f"Saved checkpoint to {ckpt_path}")

    # Quick evaluation on test set
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for batch in test_dl:
            input_ids = batch["input_ids"].to(cfg.device)
            attention_mask = batch["attention_mask"].to(cfg.device)
            traits = batch["traits"].to(cfg.device)
            emotion = batch["emotion"].to(cfg.device)
            out = model(input_ids, attention_mask)
            loss_val, parts = loss_fn(out, traits, emotion)
            total_loss += loss_val.item()
    avg_test_loss = total_loss / len(test_dl) if len(test_dl) > 0 else 0.0
    print(f"Test set ({test_n} samples) avg loss={avg_test_loss:.4f}")


if __name__ == "__main__":
    main()
