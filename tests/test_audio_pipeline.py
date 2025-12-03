"""Unit tests for audio data pipeline using synthetic generator output."""
import os
from src.data.audio_dataset import AudioDataset


def test_audio_dataset_loads():
    path = os.path.join("data", "example", "audio_sample.wav")
    ds = AudioDataset(path)
    item = ds[0]
    assert "features" in item
