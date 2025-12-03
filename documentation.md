# Project Documentation

Repository: personality_identification_based_on_text
Local path: `C:\Users\vamsh\llm_p`
Remote: https://github.com/vvkb24/personality_identification_based_on_text

Date: 2025-12-03

Summary
-------
- Purpose: A runnable demonstration repo implementing a lightweight, modular multimodal personality & emotion inference demo (text + audio + video) with training, evaluation, synthetic data, a demo server, and tests. The code is designed to run offline on synthetic data and to be easy to extend with pretrained models.
- Current state: repository scaffolding completed, synthetic example data present (`data/example/`), unit tests pass, text-only model training completed with a deterministic 80/30 split (110 samples total).

Quick reproduction
------------------
1. Activate virtual environment (PowerShell):
```
.\.venv\Scripts\Activate.ps1
```
2. (Optional) Regenerate synthetic example data:
```
python scripts/prepare_synthetic_data.py
```
3. Run unit tests:
```
python -m pytest -q
```
4. Train text model with an 80/30 split (example run, 3 epochs):
```
python -m src.train.train_text --train-size 80 --test-size 30 --epochs 3
```

Files and what they do
----------------------
Top level
- `README.md`: Project overview, quickstart and usage notes.
- `LICENSE`: MIT license.
- `.gitignore`: ignores `.venv`, `/data/`, `/models/`, and other large or environment-specific files.
- `requirements.txt`, `pyproject.toml`: Python dependency metadata.

Scripts
- `scripts/prepare_synthetic_data.py`: Generates the small example dataset under `data/example/` (JSONL text samples, small WAV, small MP4). Use to regenerate example data if you don't want to track it in git.
- `scripts/collect_consent_template.md`: Consent template used as an example for data collection ethics.

Data
- `data/README.md`: Notes about data placement and how to regenerate.
- `data/example/text_samples.jsonl`: JSON Lines — each line contains: `id`, `text`, `traits` (Big Five floats O,C,E,A,N), and `emotion` (label string: `neutral`, `happy`, `sad`). Current file has 110 samples (`s1`..`s110`).
- `data/example/audio_sample.wav`, `data/example/video_sample.mp4`: tiny media placeholders used by audio/video dataset code and tests.

Source package (`src/`)
- `src/__init__.py`: package marker.
- `src/config.py`: `Config` dataclass with defaults for `data_dir` (`data/example`), `output_dir` (`models`), `seed`, `device` (auto GPU/CPU), `batch_size`, `lr`, `epochs`, `text_max_length`, `audio_sr`, `audio_n_mels`, and `nan_check`.

Utils
- `src/utils/io_utils.py`: JSONL read/write, audio loader wrapper, and atomic checkpoint save/load helpers.
- `src/utils/audio_utils.py`: feature extraction helpers (log-mel, MFCC) with `librosa` fallback.
- `src/utils/video_utils.py`: frame extraction and optional face detection helpers (MediaPipe preferred, Haar cascade fallback).

Data modules
- `src/data/text_dataset.py`:
  - Loads the JSONL file and tokenizes text.
  - Attempts to use `transformers.AutoTokenizer('distilbert-base-uncased')` if available; otherwise falls back to a `SimpleTokenizer` (whitespace-based) so the demo runs offline.
  - Returns dicts with keys: `input_ids`, `attention_mask`, `traits` (tensor of length 5 [O,C,E,A,N]), and `emotion` (int index using `{'neutral':0,'happy':1,'sad':2}`).
- `src/data/audio_dataset.py`: builds features from audio files (log-mel spectrograms) and returns tensors for a simple `AudioModel`.
- `src/data/video_dataset.py`: extracts frames and returns frame stacks for a simple `VideoModel`.

Models
- `src/models/text_model.py`:
  - `TextModel` is a compact encoder + heads placeholder that outputs predictions for 5 trait values (regression) and an emotion class (classification).
  - `loss_fn` computes a composite loss (MSE for traits + CrossEntropy for emotion) used in training scripts.
- `src/models/audio_model.py`: simple CNN encoder that maps mel inputs to embeddings and prediction heads.
- `src/models/video_model.py`: frame-level CNN + temporal pooling producing embeddings and heads.
- `src/models/fusion_model.py`: projection heads per modality + concat fusion MLP (with optional cross-attention) and final prediction heads.

Training
- `src/train/train_text.py` (modified): Accepts CLI args `--train-size`, `--test-size`, and `--epochs`. Behavior:
  - Loads `text_samples.jsonl` from `cfg.data_dir`.
  - Performs deterministic shuffle (seed from `Config`) and creates train/test subsets based on sizes.
  - Trains for `cfg.epochs` (or `--epochs` override), saves final checkpoint to `models/text_model.pt` and evaluates on the test set printing an average loss.
  - Example run used in reproduction: `python -m src.train.train_text --train-size 80 --test-size 30 --epochs 3`.
- `src/train/train_audio.py`, `src/train/train_video.py`, `src/train/train_fusion.py`: same pattern for other modalities.

Evaluation
- `src/evaluate/*.py`: scripts that compute common metrics: MAE/MSE for trait regression, accuracy/F1 for emotion classification, and optionally produce small plots (matplotlib). These are lightweight and intended as demos.

Serving
- `src/serve/demo_server.py`: FastAPI demo server exposing `/infer` to return predicted `traits` and `emotion` from provided inputs. Minimal security and rate-limiting, not production-grade.

Tests
- `tests/test_text_model.py`: unit test that instantiates `TextModel` and checks forward pass shapes.
- `tests/test_audio_pipeline.py`: unit test that loads `data/example/audio_sample.wav` via `AudioDataset` and checks shape/processing.

Project state & git
- Local repo path: `C:\Users\vamsh\llm_p`.
- Remote pushed to: `https://github.com/vvkb24/personality_identification_based_on_text` (branch `main`).
- Note: `.venv/` and `models/` are in `.gitignore` (intentionally). Small example `data/example/*` files were force-added and pushed for convenience. For large datasets/checkpoints use Git LFS or release artifacts.

Test results (executed commands & outputs)
------------------------------------------------
- Command run (venv): `python -m pytest -q`
- Result: `2 passed` (both unit tests passed).
  - `tests/test_text_model.py`: passed.
  - `tests/test_audio_pipeline.py`: passed.

Training results (executed run)
--------------------------------
- Command run (venv):
```
python -m src.train.train_text --train-size 80 --test-size 30 --epochs 3
```
- Dataset: `data/example/text_samples.jsonl` (110 samples total). Split: 80 train, 30 test (deterministic shuffle with seed=42).
- Console output summary:
  - Epoch 1/3 finished, train avg loss = 1.4088
  - Epoch 2/3 finished, train avg loss = 1.3602
  - Epoch 3/3 finished, train avg loss = 1.3118
  - Saved checkpoint to `models\\text_model.pt`
  - Test set (30 samples) avg loss = 1.3258

Notes about these results
- The dataset is synthetic and small; reported losses are only useful for demonstration. Expect noisy and high losses compared to a real dataset and a properly tuned model.
- The model used is a lightweight stub; for realistic performance, replace the encoder with a pretrained transformer and run longer training on a larger dataset.

Reproducibility checklist
- Use the included `Config` defaults or override via CLI args.
- Use the project venv so imports work as expected: `.\\.venv\\Scripts\\Activate.ps1` then `python -m ...`.
- To retrain fully:
  1. Ensure you have enough data (the example generator is synthetic and small).
  2. Replace `TextModel` encoder with `transformers.AutoModel` and update tokenization in `TextDataset`.
  3. Increase `cfg.epochs`, add LR schedule and validation logging.

Recommended next steps
- (Short) Add evaluation metrics output (MAE for each trait + classification report for emotion) saved to a CSV in `reports/`.
- (Medium) Swap to a pretrained encoder (e.g., `distilbert-base-uncased`) and finetune — I can implement and run a short demo (requires internet to download weights).
- (Ops) Add a GitHub Actions workflow to run `pytest` on push and optionally run a smoke training job on small data.
- (Artifacts) Use Git LFS or GitHub Releases to store model checkpoints if you want them in the remote.

Where I put things
- Documentation file (this): `documentation.md` at repository root.
- Trained model: `models/text_model.pt` (written by the training run; not tracked by git by default).

If you want me to do more
- I can push the `train_text.py` changes (already committed locally). If you want, I will push any additional modifications.
- Implement a transformer-based encoder + finetune run (requires internet).
- Add evaluation CSV and per-sample diagnostics.

Contact / support
- I can run additional training, produce plots, or add CI as requested — tell me which next step to take.

---
Generated automatically on 2025-12-03 by the project maintainer assistant.
