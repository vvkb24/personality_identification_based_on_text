"""Video utilities: frame extraction, face bbox and keypoint wrappers.

This module uses OpenCV primarily and optionally MediaPipe for facial/pose
landmarks if available. It documents frame-rate and synchronization issues.
"""
from typing import List, Tuple, Dict
import cv2
import numpy as np


def extract_frames(video_path: str, max_frames: int = 32) -> List[np.ndarray]:
    """Extract up to `max_frames` uniformly sampled frames from the video.

    Important: frame rate matters when syncing audio and video. For the demo
    we take a small number of frames to keep memory low.
    """
    cap = cv2.VideoCapture(video_path)
    frames = []
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        # read sequentially (fallback)
        while len(frames) < max_frames:
            ret, frm = cap.read()
            if not ret:
                break
            frames.append(cv2.cvtColor(frm, cv2.COLOR_BGR2RGB))
    else:
        # sample indices uniformly
        indices = np.linspace(0, max(0, total - 1), min(max_frames, total)).astype(int)
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ret, frm = cap.read()
            if not ret:
                continue
            frames.append(cv2.cvtColor(frm, cv2.COLOR_BGR2RGB))
    cap.release()
    return frames


def detect_face_boxes(frames: List[np.ndarray]) -> List[Tuple[int, int, int, int]]:
    """Return a box per frame (x,y,w,h) using a simple Haar cascade fallback.

    For production, replace with MediaPipe or a robust face detector.
    Privacy note: extracting face crops is sensitive; consider storing only
    embeddings and deleting raw crops as part of data minimization.
    """
    boxes = []
    # Try MediaPipe if available for better keypoints
    try:
        import mediapipe as mp

        mp_face = mp.solutions.face_detection.FaceDetection(model_selection=0)
        for frm in frames:
            h, w, _ = frm.shape
            results = mp_face.process(cv2.cvtColor(frm, cv2.COLOR_RGB2BGR))
            if results.detections:
                d = results.detections[0]
                bbox = d.location_data.relative_bounding_box
                x = int(bbox.xmin * w)
                y = int(bbox.ymin * h)
                bw = int(bbox.width * w)
                bh = int(bbox.height * h)
                boxes.append((x, y, bw, bh))
            else:
                boxes.append((0, 0, w, h))
        return boxes
    except Exception:
        # Haar cascade fallback
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        for frm in frames:
            gray = cv2.cvtColor(frm, cv2.COLOR_RGB2GRAY)
            detected = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)
            if len(detected) > 0:
                boxes.append(tuple(detected[0]))
            else:
                h, w, _ = frm.shape
                boxes.append((0, 0, w, h))
        return boxes
