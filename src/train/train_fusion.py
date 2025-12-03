"""Train script for fusion model combining text/audio/video encoders.

For the demo we instantiate the small encoders from the model modules and run
one pass. In practice you'd load full pretrained encoders and fine-tune.
"""
import os
import torch
from ..config import get_default_config
from ..models.fusion_model import FusionModel
from ..models.text_model import TextModel
from ..models.audio_model import AudioModel
from ..models.video_model import VideoModel
from ..utils.io_utils import save_checkpoint


def main():
    cfg = get_default_config()
    os.makedirs(cfg.output_dir, exist_ok=True)
    # Build small models to produce embeddings
    text_mod = TextModel().to(cfg.device)
    audio_mod = AudioModel().to(cfg.device)
    video_mod = VideoModel().to(cfg.device)
    # For demo, create a fusion model with matching dims (we pick defaults)
    fusion = FusionModel(text_dim=64, audio_dim=64, video_dim=64, use_cross_attention=False).to(cfg.device)

    optim = torch.optim.Adam(list(fusion.parameters()), lr=cfg.lr)

    # Single dummy training step that obtains embeddings from each model
    text_in = torch.randint(0, 1000, (1, cfg.text_max_length)).to(cfg.device)
    out_text = text_mod(text_in)["traits"]  # shape (B,5) but we will reduce
    # reduce to embedding dim for demo by projecting
    text_emb = torch.randn(1, 64, device=cfg.device)
    audio_in = torch.randn(1, cfg.audio_n_mels, 20, device=cfg.device)
    out_audio = audio_mod(audio_in)
    audio_emb = torch.randn(1, 64, device=cfg.device)
    video_in = torch.randn(1, 4, 3, 64, 64, device=cfg.device)
    video_emb = torch.randn(1, 64, device=cfg.device)

    fusion_out = fusion(text_emb, audio_emb, video_emb)
    # Dummy loss
    loss = fusion_out["traits"].sum() * 0.0 + fusion_out["logits"].sum() * 0.0
    optim.zero_grad()
    loss.backward()
    optim.step()
    save_checkpoint(fusion, os.path.join(cfg.output_dir, "fusion_model.pt"), optimizer=optim)
    print("Fusion demo step complete")


if __name__ == "__main__":
    main()
