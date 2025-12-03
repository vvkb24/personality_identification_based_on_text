"""Scalable training script for tabular/questionnaire data.

Features:
- Streams CSV/TSV with `CSVQuestionnaireDataset` (memory efficient)
- Uses DataLoader with `num_workers`, `pin_memory`, and optional AMP for fp16
- CLI options to configure batch size, epochs, learning rate, checkpoint path
"""
import argparse
import os
import math
import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
from tqdm import tqdm

from ..config import get_default_config
from ..data.csv_questionnaire_dataset import CSVQuestionnaireDataset
from ..models.tabular_model import TabularModel
from ..utils.io_utils import save_checkpoint


def collate_fn(batch):
    # batch: list of examples
    # pad features to max length in batch
    feats = [b["features"] for b in batch]
    maxlen = max([f.shape[0] for f in feats]) if feats else 0
    padded = []
    for f in feats:
        if f.shape[0] < maxlen:
            pad = torch.zeros(maxlen - f.shape[0], dtype=torch.float)
            padded.append(torch.cat([f, pad], dim=0))
        else:
            padded.append(f)
    X = torch.stack(padded)
    # collect trait targets if available
    trait_list = [b.get("trait_vec") for b in batch]
    if any(t is not None for t in trait_list):
        trait_tensors = [t if t is not None else torch.zeros(5) for t in trait_list]
        Y = torch.stack(trait_tensors)
    else:
        Y = None
    # emotion
    emo_list = [b.get("emotion") for b in batch]
    if any(e is not None for e in emo_list):
        emo = [e if e is not None else 0 for e in emo_list]
        emo_t = torch.tensor(emo, dtype=torch.long)
    else:
        emo_t = None
    return {"features": X, "traits": Y, "emotion": emo_t}


def train(args):
    cfg = get_default_config()
    ds = CSVQuestionnaireDataset(args.path, sep=args.sep, text_columns=None, emotion_col=args.emotion_col)
    dl = DataLoader(ds, batch_size=args.batch_size, num_workers=args.num_workers, collate_fn=collate_fn, pin_memory=True)

    # Peek one batch to infer input dim
    it = iter(dl)
    sample = next(it)
    in_dim = sample["features"].shape[1]
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")

    model = TabularModel(input_dim=in_dim, hidden_dims=args.hidden_dims, trait_out=5, n_emotions=args.n_emotions).to(device)
    optim = Adam(model.parameters(), lr=args.lr)

    scaler = torch.cuda.amp.GradScaler(enabled=args.fp16 and device.type == "cuda")

    steps_per_epoch = args.steps_per_epoch or math.ceil(args.steps_per_epoch or 1000)

    for epoch in range(args.epochs):
        model.train()
        pbar = tqdm(dl, desc=f"Epoch {epoch+1}")
        for i, batch in enumerate(pbar):
            X = batch["features"].to(device)
            Y = batch["traits"].to(device) if batch["traits"] is not None else None
            E = batch["emotion"].to(device) if batch["emotion"] is not None else None
            optim.zero_grad()
            with torch.cuda.amp.autocast(enabled=(args.fp16 and device.type == "cuda")):
                out = model(X)
                loss = 0.0
                if Y is not None:
                    loss_fn = torch.nn.MSELoss()
                    loss = loss + loss_fn(out["traits"], Y)
                if E is not None:
                    loss_fn2 = torch.nn.CrossEntropyLoss()
                    loss = loss + loss_fn2(out["logits"], E)
            scaler.scale(loss).backward()
            scaler.step(optim)
            scaler.update()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        # Save checkpoint per epoch
        os.makedirs(args.output_dir, exist_ok=True)
        ckpt_path = os.path.join(args.output_dir, f"tabular_model_epoch{epoch+1}.pt")
        save_checkpoint(model, ckpt_path, optimizer=optim)
        print(f"Saved checkpoint: {ckpt_path}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("path", help="CSV/TSV file path")
    p.add_argument("--sep", default=None, help="separator, default auto by extension")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--fp16", action="store_true", help="enable AMP fp16 training on CUDA")
    p.add_argument("--no-cuda", action="store_true")
    p.add_argument("--output-dir", default="models")
    p.add_argument("--emotion-col", default=None)
    p.add_argument("--hidden-dims", nargs="+", type=int, default=[256, 128])
    p.add_argument("--n-emotions", type=int, default=3)
    p.add_argument("--steps-per-epoch", type=int, default=None)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)
