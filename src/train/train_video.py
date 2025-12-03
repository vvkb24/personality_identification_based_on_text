"""Train script for video-only model (demo).

Uses VideoDataset and a tiny VideoModel to illustrate training flow.
"""
import os
import torch
from torch.utils.data import DataLoader
from ..config import get_default_config
from ..data.video_dataset import VideoDataset
from ..models.video_model import VideoModel
from ..utils.io_utils import save_checkpoint


def train_one_epoch(model, dl, optim, device):
    model.train()
    total_loss = 0.0
    mse = torch.nn.MSELoss()
    ce = torch.nn.CrossEntropyLoss()
    for batch in dl:
        frames = batch["frames"].to(device)
        optim.zero_grad()
        out = model(frames)
        trait_target = torch.zeros_like(out["traits"])
        emotion_target = torch.zeros(frames.shape[0], dtype=torch.long, device=device)
        loss = mse(out["traits"], trait_target) + ce(out["logits"], emotion_target)
        loss.backward()
        optim.step()
        total_loss += loss.item()
    return total_loss / len(dl)


def main():
    cfg = get_default_config()
    os.makedirs(cfg.output_dir, exist_ok=True)
    ds = VideoDataset(os.path.join(cfg.data_dir, "video_sample.mp4"), cfg)
    dl = DataLoader(ds, batch_size=1, shuffle=False)
    model = VideoModel().to(cfg.device)
    optim = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    loss = train_one_epoch(model, dl, optim, cfg.device)
    print(f"Video epoch done, loss={loss:.4f}")
    save_checkpoint(model, os.path.join(cfg.output_dir, "video_model.pt"), optimizer=optim)


if __name__ == "__main__":
    main()
