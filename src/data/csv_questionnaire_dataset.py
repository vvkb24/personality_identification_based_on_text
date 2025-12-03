"""Iterable CSV/TSV dataset for large questionnaire-style files.

Features:
- Streams rows from a CSV/TSV file (memory-efficient for large files).
- Automatically groups item columns by prefix (EXT, EST, AGR, CSN, OPN, etc.)
  and computes mean per trait (normalizes 1-5 to 0-1). If a column named
  `emotion` exists it's parsed as the emotion label.
- Yields dicts with: `features` (float tensor of raw item values),
  `trait_vec` (float tensor of 5 traits if available/computed), and
  `emotion` (int index if present). Also supports returning `text` if
  textual columns are supplied.

Usage: instantiate with the path and pass to a DataLoader. For large files
use `DataLoader(dataset, batch_size=..., num_workers=..., pin_memory=True)`.
"""
from typing import Iterable, List, Dict, Optional
import os
import csv
import torch
from torch.utils.data import IterableDataset
import re


def _col_prefix(name: str) -> str:
    # Extract alphabetic prefix up to first digit or underscore
    m = re.match(r"([A-Za-z]+)", name)
    return m.group(1).upper() if m else name.upper()


class CSVQuestionnaireDataset(IterableDataset):
    def __init__(self, path: str, sep: str = None, text_columns: Optional[List[str]] = None,
                 trait_prefix_map: Optional[Dict[str, str]] = None, emotion_col: Optional[str] = None):
        """
        path: CSV/TSV file path
        sep: field separator (default: auto-detect by extension or comma)
        text_columns: list of column names to concatenate as `text` input (optional)
        trait_prefix_map: mapping from file prefix -> trait letter (O,C,E,A,N)
                          default mapping handles common prefixes (OPN, CSN, EXT, AGR, EST)
        emotion_col: name of emotion label column (optional)
        """
        self.path = path
        self.sep = sep or ("\t" if path.lower().endswith(".tsv") else ",")
        self.text_columns = text_columns or []
        # Default prefix -> trait mapping
        self.trait_prefix_map = trait_prefix_map or {
            "OPN": "O",
            "CSN": "C",
            "EXT": "E",
            "AGR": "A",
            "EST": "N",  # EST = emotional stability -> invert to get Neuroticism
        }
        self.emotion_col = emotion_col

    def _row_to_example(self, header: List[str], row: List[str]) -> Dict:
        d = dict(zip(header, row))
        # Build text if requested
        text = None
        if self.text_columns:
            parts = [d.get(c, "") for c in self.text_columns]
            text = " ".join([p for p in parts if p])

        # Collect raw numeric item columns and their values
        items = {}
        for k, v in d.items():
            try:
                items[k] = float(v)
            except Exception:
                # skip non-numeric
                continue

        # Group by prefix and compute mean per group
        groups = {}
        for col, val in items.items():
            prefix = _col_prefix(col)
            groups.setdefault(prefix, []).append(val)

        # Compute trait vector in order [O,C,E,A,N] if available
        trait_vec = None
        trait_vals = {"O": None, "C": None, "E": None, "A": None, "N": None}
        for prefix, vals in groups.items():
            if prefix in self.trait_prefix_map:
                trait_key = self.trait_prefix_map[prefix]
                # Normalize 1-5 -> 0-1 if values appear in that range, else use min-max later
                meanv = sum(vals) / len(vals)
                # If the prefix is EST (emotional stability), invert to get Neuroticism
                if prefix == "EST":
                    norm = (meanv - 1.0) / 4.0
                    trait_vals["N"] = 1.0 - norm
                else:
                    trait_vals[trait_key] = (meanv - 1.0) / 4.0

        # Build trait_vec if any were found
        if any(v is not None for v in trait_vals.values()):
            # Fill missing with 0.5 (neutral)
            vec = [trait_vals["O"] or 0.5, trait_vals["C"] or 0.5, trait_vals["E"] or 0.5,
                   trait_vals["A"] or 0.5, trait_vals["N"] or 0.5]
            trait_vec = torch.tensor(vec, dtype=torch.float)

        # Feature vector: use all numeric items in header order for predictability
        numeric_cols = [c for c in header if re.match(r"^[A-Za-z].*", c) and c in items]
        feature_vec = [items[c] for c in numeric_cols]
        feature_tensor = torch.tensor(feature_vec, dtype=torch.float) if feature_vec else torch.empty(0)

        # Emotion label
        emotion_idx = None
        if self.emotion_col and self.emotion_col in d:
            try:
                emotion_idx = int(d[self.emotion_col])
            except Exception:
                # leave None
                pass

        example = {"features": feature_tensor, "trait_vec": trait_vec, "text": text, "emotion": emotion_idx}
        return example

    def __iter__(self) -> Iterable[Dict]:
        # Stream rows
        with open(self.path, "r", encoding="utf-8") as f:
            reader = csv.reader(f, delimiter=self.sep)
            header = next(reader)
            for row in reader:
                if not any(row):
                    continue
                yield self._row_to_example(header, row)
