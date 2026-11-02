"""Off-the-shelf image classifier: torchvision MobileNetV3-Small with ImageNet weights, on CPU.

No training happens in this project. The weights (`MobileNet_V3_Small_Weights.IMAGENET1K_V1`, about 10 MB)
come from the torchvision model zoo (`download.pytorch.org`) and are cached under `TORCH_HOME`. The
Docker image downloads them at build time, so the running container needs no network and no mounts.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from PIL import Image
from torchvision.models.mobilenetv3 import MobileNet_V3_Small_Weights, mobilenet_v3_small

WEIGHTS = MobileNet_V3_Small_Weights.IMAGENET1K_V1


@dataclass(frozen=True)
class Prediction:
    """Softmax over the 1000 ImageNet classes plus their names (same order)."""

    probs: np.ndarray
    categories: tuple[str, ...]


class ImageClassifier:
    def __init__(self) -> None:
        self.model = mobilenet_v3_small(weights=WEIGHTS).eval()
        self.preprocess = WEIGHTS.transforms()
        self.categories: tuple[str, ...] = tuple(WEIGHTS.meta["categories"])

    def predict_image(self, image: Image.Image) -> Prediction:
        batch = self.preprocess(image.convert("RGB")).unsqueeze(0)
        with torch.inference_mode():
            logits = self.model(batch)
        probs = torch.softmax(logits, dim=1)[0].to(torch.float64).numpy()
        return Prediction(probs=probs, categories=self.categories)
