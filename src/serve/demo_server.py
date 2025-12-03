"""Simple FastAPI demo server to perform inference on text/audio/video inputs.

The server uses the lightweight model stubs to run quick inference on incoming
requests. The `/infer` endpoint accepts JSON with `text`, or base64 audio, or
path to video, and returns personality scores, emotion label, and confidence.
"""
from typing import Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import base64
import io
import numpy as np
import torch
from ..models.text_model import TextModel
from ..config import get_default_config
from ..utils.io_utils import read_audio


app = FastAPI(title="Multimodal Demo Server")
cfg = get_default_config()

# Simple request schema
class InferRequest(BaseModel):
    text: Optional[str] = None
    audio_b64: Optional[str] = None  # base64 encoded WAV file
    video_path: Optional[str] = None  # path on server (demo only)


# Load small models (stubs) for demo
text_model = TextModel()
text_model.eval()


@app.post("/infer")
def infer(req: InferRequest):
    # Basic routing: prefer text, then audio, then video
    if req.text:
        # Very small tokenization: whitespace -> ids (matches TextDataset fallback)
        toks = req.text.strip().split()
        ids = [i + 2 for i in range(min(len(toks), cfg.text_max_length))]
        # pad
        pad = [0] * (cfg.text_max_length - len(ids))
        input_ids = torch.tensor([ids + pad], dtype=torch.long)
        out = text_model(input_ids)
        traits = torch.sigmoid(out["traits"]).detach().numpy().tolist()[0]
        logits = out["logits"].detach().numpy()[0]
        emotion_idx = int(np.argmax(logits))
        emotion = ["neutral", "happy", "sad"][emotion_idx]
        conf = float(torch.softmax(torch.tensor(logits), dim=0).max())
        return {"traits": traits, "emotion": emotion, "confidence": conf}

    if req.audio_b64:
        # Decode and run lightweight audio path (demo)
        try:
            b = base64.b64decode(req.audio_b64)
            bio = io.BytesIO(b)
            # Note: read_audio expects a path; for demo this may fail depending on soundfile support
            data, sr = read_audio(bio, target_sr=cfg.audio_sr)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid audio: {e}")
        # produce dummy response
        return {"traits": [0.5] * 5, "emotion": "neutral", "confidence": 0.6}

    if req.video_path:
        # For demo, do not accept arbitrary uploads; require server-local path
        return {"traits": [0.5] * 5, "emotion": "neutral", "confidence": 0.6}

    raise HTTPException(status_code=400, detail="No input provided; send text, audio_b64 or video_path")
