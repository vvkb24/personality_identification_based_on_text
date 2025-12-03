"""I/O utilities: JSONL read/write, audio read/write wrappers, checkpoint helpers.

This module centralizes common I/O routines and documents common pitfalls such
as sample rate mismatches, stereo vs mono, and atomic checkpoint saves.
"""
import json
import os
from typing import Iterable, Dict, Any
import soundfile as sf
import torch


def read_jsonl(path: str) -> Iterable[Dict[str, Any]]:
    """Yield JSON objects from a JSONL file.

    Inputs:
    - path: path to JSONL

    Outputs:
    - yields dicts

    Note: JSONL is preferable for streaming large datasets because each record
    is self-contained per line and can be read without loading entire file.
    """
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def write_jsonl(path: str, records: Iterable[Dict[str, Any]]):
    """Write records to JSONL atomically.

    We write to a temporary file and move it into place to avoid partial writes.
    """
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    os.replace(tmp, path)


def read_audio(path: str, target_sr: int = None):
    """Read audio with soundfile and optionally resample.

    Returns numpy array (mono) and sample rate.

    Pitfalls:
    - Many datasets use different sample rates (8k, 16k, 44.1k). Pick a
      canonical `target_sr` (speech models often use 16k).
    - Some files are stereo; we convert to mono by averaging channels.
    - Use a stable resampler (librosa or torchaudio) when resampling.
    """
    data, sr = sf.read(path)
    # Convert to mono if needed
    if data.ndim > 1:
        data = data.mean(axis=1)
    if target_sr is not None and sr != target_sr:
        try:
            import librosa

            data = librosa.resample(data, orig_sr=sr, target_sr=target_sr)
            sr = target_sr
        except Exception:
            # If librosa is not available, leave as-is and warn
            print("Warning: librosa unavailable for resampling; returning original sr")
    return data, sr


def save_checkpoint(model: torch.nn.Module, path: str, optimizer: torch.optim.Optimizer = None):
    """Save model state dict and optimizer in an atomic way.

    Using `tmp` file and `os.replace` to ensure partial writes do not corrupt
    checkpoints. This is critical in production where power/IO failures can
    otherwise leave a broken file.
    """
    tmp = path + ".tmp"
    state = {"model_state": model.state_dict()}
    if optimizer is not None:
        state["optimizer_state"] = optimizer.state_dict()
    torch.save(state, tmp)
    os.replace(tmp, path)


def load_checkpoint(path: str, model: torch.nn.Module, optimizer: torch.optim.Optimizer = None):
    """Load checkpoint into model/optimizer if keys match.

    Returns: dict with saved metadata (if present).
    """
    state = torch.load(path, map_location="cpu")
    model.load_state_dict(state["model_state"])
    if optimizer is not None and "optimizer_state" in state:
        optimizer.load_state_dict(state["optimizer_state"])
    return state
