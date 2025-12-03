"""Run evaluation for the text model and print/save metrics.

This script recreates the deterministic 80/30 split used during training
and evaluates the saved checkpoint at `models/text_model.pt`.
"""
import os
import random
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from ..config import get_default_config
from ..data.text_dataset import TextDataset
from ..models.text_model import TextModel
from ..utils.io_utils import load_checkpoint
from sklearn.metrics import mean_absolute_error, mean_squared_error, accuracy_score, f1_score, classification_report
import matplotlib.pyplot as plt


def main(train_size: int = 80, test_size: int = 30, ckpt_path: str = "models/text_model.pt"):
    cfg = get_default_config()
    device = cfg.device

    ds = TextDataset(os.path.join(cfg.data_dir, "text_samples.jsonl"), cfg)
    total = len(ds)
    if train_size + test_size > total:
        raise ValueError("Requested split larger than dataset")

    rng = random.Random(cfg.seed)
    indices = list(range(total))
    rng.shuffle(indices)
    test_idx = indices[train_size: train_size + test_size]
    test_ds = Subset(ds, test_idx)
    dl = DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=False)

    model = TextModel()
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found at {ckpt_path}")
    load_checkpoint(ckpt_path, model)
    model.to(device)
    model.eval()

    traits_true = []
    traits_pred = []
    emotions_true = []
    emotions_pred = []

    with torch.no_grad():
        for batch in dl:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            t_true = batch["traits"].numpy()
            e_true = batch["emotion"].numpy()
            out = model(input_ids, attention_mask)
            t_pred = out["traits"].cpu().numpy()
            logits = out["logits"].cpu().numpy()
            e_pred = np.argmax(logits, axis=1)

            traits_true.append(t_true)
            traits_pred.append(t_pred)
            emotions_true.append(e_true)
            emotions_pred.append(e_pred)

    traits_true = np.vstack(traits_true)
    traits_pred = np.vstack(traits_pred)
    emotions_true = np.concatenate(emotions_true)
    emotions_pred = np.concatenate(emotions_pred)

    # Regression metrics
    mae_all = mean_absolute_error(traits_true, traits_pred)
    mse_all = mean_squared_error(traits_true, traits_pred)
    print(f"Traits MAE (all) = {mae_all:.4f}")
    print(f"Traits MSE (all) = {mse_all:.4f}")

    # Per-trait MAE
    per_trait_mae = mean_absolute_error(traits_true, traits_pred, multioutput='raw_values')
    for i, v in enumerate(per_trait_mae):
        print(f"Trait {i} MAE = {v:.4f}")

    # Classification metrics
    acc = accuracy_score(emotions_true, emotions_pred)
    f1 = f1_score(emotions_true, emotions_pred, average='macro')
    print(f"Emotion Accuracy = {acc:.4f}")
    print(f"Emotion F1 (macro) = {f1:.4f}")
    print("Classification report:\n", classification_report(emotions_true, emotions_pred))

    # Save a scatter plot for trait 0
    plt.figure(figsize=(5, 5))
    plt.scatter(traits_true[:, 0], traits_pred[:, 0])
    plt.xlabel("True Trait 0 (O)")
    plt.ylabel("Pred Trait 0 (O)")
    plt.title("Trait 0: true vs pred")
    plt.grid(True)
    out_png = "eval_trait0.png"
    plt.savefig(out_png)
    print(f"Saved plot {out_png}")


if __name__ == "__main__":
    main()
