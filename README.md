# Multimodal Personality & Emotion Inference (Demo)

Project overview
-----------------
This repository contains a minimal, production-oriented demo of a multimodal (text + audio + video) personality and emotion inference system. The code is intentionally small and well-documented so beginners can run everything locally on synthetic data without external downloads.

Key features
- Lightweight per-modality models (stubs) that are easy to replace with pretrained checkpoints.
- Synthetic demo data to train and evaluate quickly.
- Simple fusion model (concatenate + optional cross-attention).
- FastAPI demo server to run inference locally.

Architecture (ASCII diagram)

 Text Input  --> Text Encoder  -----\\
                                      --> Fusion --> Heads (Big5 regression + Emotion classification)
 Audio Input --> Audio Encoder -----/
 Video Input --> Video Encoder ----/

Quickstart
----------
Prerequisites: Python 3.10+, pip.

1. Create and activate a virtual environment (recommended):

```pwsh
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -U pip
```

2. Install dependencies:

```pwsh
pip install -r requirements.txt
```

3. Prepare synthetic data (creates `data/example`):

```pwsh
python scripts/prepare_synthetic_data.py
```

4. Train small text model for 1 epoch (demo):

```pwsh
python src/train/train_text.py
```

5. Run the demo server:

```pwsh
pip install uvicorn fastapi
uvicorn src.serve.demo_server:app --reload --port 8000
```

Expected outputs
----------------
- Personality: Big-Five scores (continuous regression values in range ~[0,1]) for traits: O, C, E, A, N. These are returned as a 5-element vector.
- Emotion: Discrete emotion label (e.g., neutral, happy, sad) with confidence.

Datasets
--------
This repo uses synthetic example data included in `data/example/`. Replace these with real datasets carefully. See `data/README.md` for instructions.

Ethics & Privacy
-----------------
Personality and emotion inference can be harmful if misused. Important best-practices:
- Obtain informed consent from participants (see `scripts/collect_consent_template.md`).
- Minimize collected data and store it securely.
- Use human oversight for decisions that affect people.
- Avoid deployment on sensitive populations without extensive validation.

TODOs for developers
- Replace lightweight stubs with pretrained models (see many TODO markers).
- Add real datasets and update data loaders.

License
-------
MIT. See `LICENSE`.
