"""Encode video frames for multimodal generation."""

import base64
from pathlib import Path

import cv2
import numpy as np

from utils.llm_gpt import get_response


def generate_messages(images, prompt):
    """Build OpenAI-style multimodal messages from images + a text prompt.

    images: np.ndarray, path, directory, or iterable of these.
    Images are resized to half resolution to stay within request size limits.
    """
    if isinstance(images, (str, Path, np.ndarray)):
        images = [images]

    imgs = []
    for item in images:
        if isinstance(item, np.ndarray):
            imgs.append(item)
            continue
        p = Path(item)
        if p.is_dir():
            paths = sorted([x for x in p.iterdir() if x.suffix.lower() in [".jpg", ".jpeg"]])
            for img_path in paths:
                img = cv2.imread(str(img_path))
                if img is None:
                    raise ValueError(f"Could not read image: {img_path}")
                imgs.append(img)
        else:
            img = cv2.imread(str(p))
            if img is None:
                raise ValueError(f"Could not read image: {p}")
            imgs.append(img)

    if not imgs:
        raise ValueError("No images provided.")

    base64_frames = []
    for img in imgs:
        h, w = img.shape[:2]
        img = cv2.resize(img, (w // 2, h // 2), interpolation=cv2.INTER_AREA)
        success, buffer = cv2.imencode(".jpg", img)
        if not success:
            raise ValueError("Failed to encode image array to JPG.")
        base64_frames.append(base64.b64encode(buffer).decode("utf-8"))

    content = [
        {"type": "text", "text": prompt},
        *[
            {"type": "image_url",
             "image_url": {"url": f"data:image/jpeg;base64,{frame}"}}
            for frame in base64_frames
        ],
    ]
    return [{"role": "user", "content": content}]
