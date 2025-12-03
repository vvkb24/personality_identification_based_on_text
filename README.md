# Multimodal Personality & Emotion Inference — Demo

This repository is a compact, runnable demo that shows how to build a modular
multimodal (text + audio + video) system that predicts personality (Big Five)
and a simple emotion label. It's intended for learning, experimentation, and
rapid prototyping. The code is small, well-documented, and runs on synthetic
data so you can reproduce results offline.

Highlights
 - Modular per-modality encoders (text/audio/video) and a fusion stage.
 - Training & evaluation scripts, lightweight unit tests, and a demo FastAPI server.
 - Synthetic data generator (no external downloads required).

Repository layout (important files)
 - `src/` — core package (configs, utils, datasets, models, train, evaluate, serve)
 - `data/example/` — synthetic example dataset (JSONL + small media files)
 - `scripts/prepare_synthetic_data.py` — regenerate the example dataset
 - `tests/` — lightweight unit tests
 - `models/` — where checkpoints are saved (ignored by default)
 - `documentation.md` — detailed project documentation (auto-generated)

Quickstart (Windows PowerShell)
1) Create & activate virtualenv (recommended):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

2) Install dependencies:

```powershell
pip install -r requirements.txt
```

3) (Optional) Regenerate synthetic example data:

```powershell
python scripts/prepare_synthetic_data.py
```

4) Run unit tests (recommended before training):

```powershell
python -m pytest -q
```

5) Train the text-only demo model (example: 80/30 split, 3 epochs):

```powershell
python -m src.train.train_text --train-size 80 --test-size 30 --epochs 3
```

6) Run the demo server locally (after installing FastAPI & Uvicorn):

```powershell
pip install fastapi uvicorn
uvicorn src.serve.demo_server:app --reload --port 8000
```

Notes on running
 - Always run training/eval from the repository root and with the project
     virtual environment active so `src` imports resolve correctly (use
     `python -m ...`).
 - `data/` and `models/` are in `.gitignore`. The small example data has been
     force-added to the repository for convenience; for larger datasets use Git LFS
     or external storage.

Reproducible commands used in this demo
 - Unit tests: `python -m pytest -q` → `2 passed`
 - Training example: `python -m src.train.train_text --train-size 80 --test-size 30 --epochs 3`
     - Output (example):
         - Epoch 1 train avg loss = 1.4088
         - Epoch 2 train avg loss = 1.3602
         - Epoch 3 train avg loss = 1.3118
         - Test set (30 samples) avg loss = 1.3258

Where to look next (developer tips)
 - Replace `TextModel` with a pretrained transformer (e.g. `distilbert`) in
     `src/models/text_model.py` and switch tokenization in
     `src/data/text_dataset.py` to `transformers.AutoTokenizer` for better results.
 - Add evaluation reporting in `src/evaluate/` to save MAE per trait and
     classification metrics (precision/recall/F1) for emotion.
 - Use GitHub Actions to run `pytest` on push and optionally run a small
     training smoke test.

Ethics and privacy
 - Predicting personality and emotion can be sensitive — obtain informed
     consent, aggregate and anonymize results, add human oversight, and avoid
     using models for high-stakes decisions without validation.

License
 - This repository is released under the MIT license. See `LICENSE` for details.

If you want, I can update the README further with one of the following:
 - an expanded `How it works` section with diagrams,
 - a sample inference request/response for the demo server,
 - or a short migration guide for swapping in `transformers`.
Tell me which and I'll update it.

## Metrics & Results (example run)

These metrics were produced by evaluating the saved text-only checkpoint `models/text_model.pt` on the deterministic 80/30 test split (30 samples) produced from `data/example/text_samples.jsonl`.

Regression (Big-Five traits):

- Overall MAE: 0.3713
- Overall MSE: 0.1981
- Per-trait MAE:
    - Trait 0 (O): 0.4193
    - Trait 1 (C): 0.4003
    - Trait 2 (E): 0.3509
    - Trait 3 (A): 0.4308
    - Trait 4 (N): 0.2552

Classification (emotion):

- Accuracy: 0.2000 (6 / 30)
- Macro F1: 0.1111
- Notes: The classifier mostly predicted the `sad` class, yielding high recall for that label and zero precision for others. This behavior is expected on a tiny synthetic dataset with a simple model.

Artifacts produced by the evaluation run:

- Checkpoint: `models/text_model.pt` (saved by the training script)
- Plot: `eval_trait0.png` (scatter of true vs predicted for Trait 0)

You can reproduce the evaluation with:

```powershell
.\.venv\Scripts\Activate.ps1
python -m src.evaluate.run_eval_text
```

