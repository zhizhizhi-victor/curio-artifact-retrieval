"""Frozen image-only backbone for reference matching; no text embedding API."""

import os
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps
import torch
from transformers import AutoImageProcessor, AutoModel


DINO_ID = "facebook/dinov2-small"
DINO_REVISION = "ed25f3a31f01632728cabb09d1542f84ab7b0056"


def _default_cache_dir() -> str | None:
    if os.getenv("HF_HUB_CACHE") or os.getenv("HF_HOME") or os.getenv("TRANSFORMERS_CACHE"):
        return None
    bundled = Path(__file__).resolve().parents[3] / "models" / "huggingface" / "hub"
    return str(bundled) if bundled.is_dir() else None


class DinoEncoder:
    def __init__(self, device="auto"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if device == "auto" else torch.device(device)
        torch.set_num_threads(min(4, torch.get_num_threads()))
        cache_dir = _default_cache_dir()
        pretrained_kwargs = {"cache_dir": cache_dir} if cache_dir else {}
        self.processor = AutoImageProcessor.from_pretrained(
            DINO_ID, revision=DINO_REVISION, use_fast=False, **pretrained_kwargs
        )
        self.model = AutoModel.from_pretrained(
            DINO_ID, revision=DINO_REVISION, **pretrained_kwargs
        ).to(self.device).eval()

    @property
    def metadata(self):
        return {"encoder": "dinov2", "model_id": DINO_ID, "revision": DINO_REVISION,
                "device": str(self.device), "feature": "last_hidden_state_cls", "normalization": "l2"}

    def encode_images(self, paths, batch_size=16):
        vectors = []
        with torch.inference_mode():
            for start in range(0, len(paths), batch_size):
                images = []
                for value in paths[start:start + batch_size]:
                    if isinstance(value, Image.Image):
                        images.append(ImageOps.exif_transpose(value).convert("RGB"))
                    else:
                        with Image.open(Path(value)) as source:
                            images.append(ImageOps.exif_transpose(source).convert("RGB"))
                inputs = self.processor(images=images, return_tensors="pt").to(self.device)
                features = self.model(**inputs).last_hidden_state[:, 0]
                vectors.append(torch.nn.functional.normalize(features.float(), dim=-1).cpu().numpy())
        return np.concatenate(vectors).astype(np.float32) if vectors else np.empty((0, 384), dtype=np.float32)
