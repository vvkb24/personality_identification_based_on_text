"""Dataset for text samples using transformers tokenizer with fallback.

This dataset reads a JSONL file of examples and tokenizes text. The code
attempts to use a `transformers` tokenizer (e.g., distilbert) but falls back
to a simple whitespace tokenizer to ensure the demo runs offline.
"""
from typing import List, Dict, Any, Optional
import os
import json
import torch
from torch.utils.data import Dataset
from ..utils.io_utils import read_jsonl
from ..config import Config, get_default_config


class SimpleTokenizer:
    """A minimal whitespace tokenizer used as a fallback.

    It maps tokens to integer ids and pads/truncates to `max_length`.
    """

    def __init__(self, max_length=128):
        self.vocab = {"[PAD]": 0, "[UNK]": 1}
        self.max_length = max_length

    def encode(self, text: str) -> List[int]:
        tokens = text.strip().split()
        ids = []
        for t in tokens:
            if t not in self.vocab:
                self.vocab[t] = len(self.vocab)
            ids.append(self.vocab[t])
        # pad/truncate
        ids = ids[: self.max_length]
        padding = [0] * (self.max_length - len(ids))
        return ids + padding


class TextDataset(Dataset):
    """Torch Dataset that yields token ids, attention masks, and labels.

    Expected input JSONL fields per line: id, text, traits (dict of 5 floats),
    emotion (label string). Traits are used for Big-Five regression; emotion
    for classification.
    """

    def __init__(self, path: str, config: Optional[Config] = None):
        self.config = config or get_default_config()
        self.path = path
        self.examples = list(read_jsonl(path))
        # Try to load a transformers tokenizer; fallback to SimpleTokenizer
        try:
            from transformers import AutoTokenizer

            self.tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
        except Exception:
            print("Warning: transformers unavailable or offline, using SimpleTokenizer")
            self.tokenizer = SimpleTokenizer(max_length=self.config.text_max_length)

        # Build label mapping for emotions
        self.emotion2idx = {"neutral": 0, "happy": 1, "sad": 2}

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        ex = self.examples[idx]
        text = ex.get("text", "")
        # Tokenize using the selected tokenizer interface
        if hasattr(self.tokenizer, "encode_plus"):
            tok = self.tokenizer.encode_plus(text, max_length=self.config.text_max_length,
                                             padding="max_length", truncation=True, return_tensors="pt")
            input_ids = tok["input_ids"].squeeze(0)
            attention_mask = tok["attention_mask"].squeeze(0)
        else:
            ids = self.tokenizer.encode(text)
            input_ids = torch.tensor(ids, dtype=torch.long)
            attention_mask = (input_ids != 0).long()

        traits = ex.get("traits", {"O": 0, "C": 0, "E": 0, "A": 0, "N": 0})
        trait_vec = torch.tensor([traits["O"], traits["C"], traits["E"], traits["A"], traits["N"]], dtype=torch.float)
        emotion = ex.get("emotion", "neutral")
        emotion_idx = self.emotion2idx.get(emotion, 0)

        return {"input_ids": input_ids, "attention_mask": attention_mask, "traits": trait_vec, "emotion": emotion_idx}
