"""One normalized embedding contract for both baseline and multilingual encoders."""

from __future__ import annotations

import hashlib
import os
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps
from transformers import AutoModel, AutoProcessor

MODEL_IDS = {
    "clip": "openai/clip-vit-base-patch32",
    "siglip": "google/siglip2-base-patch16-224",
}
MODEL_REVISIONS = {
    "clip": "3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268",
    "siglip": "75de2d55ec2d0b4efc50b3e9ad70dba96a7b2fa2",
}


def _default_cache_dir() -> str | None:
    if os.getenv("HF_HUB_CACHE") or os.getenv("HF_HOME") or os.getenv("TRANSFORMERS_CACHE"):
        return None
    bundled = Path(__file__).resolve().parents[3] / "models" / "huggingface" / "hub"
    return str(bundled) if bundled.is_dir() else None


class Encoder:
    def __init__(self, name: str = "siglip", device: str = "auto", adapter_path=None):
        if name not in MODEL_IDS:
            raise ValueError(f"Unknown encoder {name!r}; choose {list(MODEL_IDS)}")
        self.name = name
        self.model_id = MODEL_IDS[name]
        eff_device = os.getenv("CURIO_DEVICE", device) if device == "auto" else device
        self.device = (
            torch.device("cuda" if torch.cuda.is_available() else "cpu")
            if eff_device == "auto"
            else torch.device(eff_device)
        )
        # A few CPU threads avoid BLAS oversubscription on desktop machines.
        torch.set_num_threads(min(4, torch.get_num_threads()))
        cache_dir = _default_cache_dir()
        pretrained_kwargs = {"cache_dir": cache_dir} if cache_dir else {}
        self.processor = AutoProcessor.from_pretrained(
            self.model_id, revision=MODEL_REVISIONS[name], use_fast=False, **pretrained_kwargs
        )
        self.model = (
            AutoModel.from_pretrained(
                self.model_id, revision=MODEL_REVISIONS[name], **pretrained_kwargs
            )
            .to(self.device)
            .eval()
        )
        self.revision = getattr(self.model.config, "_commit_hash", None)
        self.adapter_path = str(adapter_path) if adapter_path else None
        self.adapter_hash = None
        if adapter_path:
            from peft import PeftModel

            resolved_adapter = Path(str(adapter_path).replace("\\", "/"))
            adapter_file = resolved_adapter / "adapter_model.safetensors"
            self.adapter_hash = hashlib.sha256(adapter_file.read_bytes()).hexdigest()
            # Merge for inference so the existing get_*_features API remains unchanged.
            self.model = PeftModel.from_pretrained(self.model, str(resolved_adapter)).merge_and_unload()
            self.model.eval()

    @property
    def metadata(self):
        return {
            "encoder": self.name,
            "model_id": self.model_id,
            "revision": self.revision,
            "adapter_path": self.adapter_path,
            "adapter_hash": self.adapter_hash,
            "device": str(self.device),
            "normalization": "l2",
        }

    def autocast(self):
        return (
            torch.autocast("cuda", dtype=torch.float16)
            if self.device.type == "cuda"
            else nullcontext()
        )

    def image_inputs(self, images: list[Image.Image]):
        return self.processor(images=images, return_tensors="pt").to(self.device)

    def text_inputs(self, texts: list[str]):
        args = {"text": texts, "return_tensors": "pt", "truncation": True}
        if self.name == "clip":
            args.update(padding=True, max_length=77)
        else:
            args.update(padding="max_length", max_length=64)
        return self.processor(**args).to(self.device)

    def encode_images(self, paths, batch_size=16) -> np.ndarray:
        outputs = []
        with torch.inference_mode():
            for start in range(0, len(paths), batch_size):
                images = []
                for path in paths[start : start + batch_size]:
                    if isinstance(path, Image.Image):
                        images.append(ImageOps.exif_transpose(path).convert("RGB"))
                    else:
                        with Image.open(path) as image:
                            images.append(ImageOps.exif_transpose(image).convert("RGB"))
                with self.autocast():
                    features = self.model.get_image_features(**self.image_inputs(images))
                features = torch.nn.functional.normalize(features.float(), dim=-1)
                outputs.append(features.cpu().numpy())
        if not outputs:
            dim = getattr(self.model.config, "projection_dim", 768)
            return np.empty((0, dim), dtype=np.float32)
        return np.concatenate(outputs).astype(np.float32)

    def encode_texts(self, texts: list[str], batch_size=32) -> np.ndarray:
        outputs = []
        with torch.inference_mode():
            for start in range(0, len(texts), batch_size):
                with self.autocast():
                    features = self.model.get_text_features(
                        **self.text_inputs(texts[start : start + batch_size])
                    )
                outputs.append(
                    torch.nn.functional.normalize(features.float(), dim=-1).cpu().numpy()
                )
        if not outputs:
            dim = getattr(self.model.config, "projection_dim", 768)
            return np.empty((0, dim), dtype=np.float32)
        return np.concatenate(outputs).astype(np.float32)
