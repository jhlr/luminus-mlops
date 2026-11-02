"""Classificador de imagens pronto: torchvision MobileNetV3-Small com pesos ImageNet, em CPU.

Nada e treinado neste projeto. Os pesos (`MobileNet_V3_Small_Weights.IMAGENET1K_V1`, cerca de 10 MB)
vem do model zoo do torchvision (`download.pytorch.org`) e ficam em cache em `TORCH_HOME`. A imagem
Docker baixa os pesos no build, entao o container em execucao nao precisa de rede nem de montagens.
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
    """Softmax sobre as 1000 classes do ImageNet mais os nomes delas (mesma ordem)."""

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
