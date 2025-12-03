"""Create small synthetic dataset for demo: text, audio (WAV), and video (MP4).

This script writes files into `data/example/` so the rest of the demo can run
without external downloads. The audio is a sine wave with simple prosody.
The video is a short MP4 created with OpenCV composed of blank frames with
overlayed text. These are intentionally small and fast to generate.

Run: `python scripts/prepare_synthetic_data.py`
"""
import os
import json
import math
import numpy as np
import soundfile as sf
import cv2

OUT_DIR = os.path.join("data", "example")
os.makedirs(OUT_DIR, exist_ok=True)


def write_text_examples(path):
    examples = [
        {"id": "s1", "text": "I love meeting new people and exploring new ideas.",
         "traits": {"O": 0.8, "C": 0.6, "E": 0.9, "A": 0.7, "N": 0.2}, "emotion": "happy"},
        {"id": "s2", "text": "I prefer routines and organized environments.",
         "traits": {"O": 0.3, "C": 0.9, "E": 0.2, "A": 0.6, "N": 0.4}, "emotion": "neutral"},
    ]
    with open(path, "w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex) + "\n")


def write_audio(path, duration=2.0, sr=16000):
    # Create a short sine wave with a simple envelope to mimic prosody.
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    freq = 220.0
    signal = 0.3 * np.sin(2 * math.pi * freq * t)
    # Add a slow amplitude modulation to mimic prosody
    signal *= (0.5 + 0.5 * np.sin(2 * math.pi * 2.0 * t))
    sf.write(path, signal.astype(np.float32), sr)


def write_video(path, width=320, height=240, fps=15, seconds=2):
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(path, fourcc, fps, (width, height))
    frames = fps * seconds
    for i in range(frames):
        img = np.zeros((height, width, 3), dtype=np.uint8) + 60
        text = f"Frame {i+1}"
        cv2.putText(img, text, (20, height // 2), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (255, 255, 255), 2, cv2.LINE_AA)
        out.write(img)
    out.release()


def main():
    print("Generating synthetic data in", OUT_DIR)
    write_text_examples(os.path.join(OUT_DIR, "text_samples.jsonl"))
    write_audio(os.path.join(OUT_DIR, "audio_sample.wav"))
    write_video(os.path.join(OUT_DIR, "video_sample.mp4"))
    print("Done.")


if __name__ == "__main__":
    main()
