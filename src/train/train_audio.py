"""Train script for audio-only model (demo).

Similar structure to text trainer; uses AudioDataset and a light-weight model.
"""
import os
import torch
from torch.utils.data import DataLoader
from ..config import get_default_config
from ..data.audio_dataset import AudioDataset
from ..models.audio_model import AudioModel
from ..utils.io_utils import save_checkpoint


def train_one_epoch(model, dl, optim, device):
    model.train()
    total_loss = 0.0
    mse = torch.nn.MSELoss()
    ce = torch.nn.CrossEntropyLoss()
    for batch in dl:
        features = batch["features"].to(device)
        # ensure shape (B, C, T)
        if features.ndim == 2:
            features = features.unsqueeze(0)
        optim.zero_grad()
        out = model(features)
        # dummy targets for demo
        trait_target = torch.zeros(out["traits"].shape, device=device)
        emotion_target = torch.zeros(features.shape[0], dtype=torch.long, device=device)
        loss = mse(out["traits"], trait_target) + ce(out["logits"], emotion_target)
        loss.backward()
        optim.step()
        total_loss += loss.item()
    return total_loss / len(dl)


def main():
    cfg = get_default_config()
    os.makedirs(cfg.output_dir, exist_ok=True)
    ds = AudioDataset(os.path.join(cfg.data_dir, "audio_sample.wav"), cfg)
    dl = DataLoader(ds, batch_size=1, shuffle=False)
    model = AudioModel(n_mels=cfg.audio_n_mels).to(cfg.device)
    optim = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    loss = train_one_epoch(model, dl, optim, cfg.device)
    print(f"Audio epoch done, loss={loss:.4f}")
    save_checkpoint(model, os.path.join(cfg.output_dir, "audio_model.pt"), optimizer=optim)


if __name__ == "__main__":
    main()
