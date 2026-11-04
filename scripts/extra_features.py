"""Tres variaveis extras de imagem (atividade de MLOps): colorfulness, edge_density e dominant_color.

Complementam as medias L/R/G/B do curso. Uso:
    python scripts/extra_features.py img1.jpg img2.jpg img3.jpg
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from PIL import Image

EDGE_THRESHOLD = 25.0  # nivel de cinza por pixel (gradiente Sobel normalizado)
# (nome, ate qual matiz em graus, exclusivo); o ultimo "red" fecha o circulo
HUE_NAMES: tuple[tuple[str, float], ...] = (
    ("red", 15),
    ("orange", 40),
    ("yellow", 70),
    ("green", 160),
    ("cyan", 200),
    ("blue", 255),
    ("purple", 290),
    ("pink", 345),
    ("red", 360),
)
COLOR_NAMES: tuple[str, ...] = ("black", "white", "gray", "red", "orange", "yellow", "green", "cyan", "blue", "purple", "pink")


def colorfulness(rgb: np.ndarray) -> float:
    """Metrica de Hasler e Suesstrunk (2003): quao vivas sao as cores. 0 = cinza puro."""
    r, g, b = (rgb[..., i].astype(np.float64) for i in range(3))
    rg = r - g
    yb = 0.5 * (r + g) - b
    return float(np.hypot(rg.std(), yb.std()) + 0.3 * np.hypot(rg.mean(), yb.mean()))


def edge_density(rgb: np.ndarray, threshold: float = EDGE_THRESHOLD) -> float:
    """Fracao de pixels com gradiente de Sobel (sobre a luminancia) acima de `threshold`: 0 a 1."""
    luma = np.asarray(Image.fromarray(rgb).convert("L"), dtype=np.float64)
    if min(luma.shape) < 3:
        return 0.0
    gx = (luma[:-2, 2:] + 2 * luma[1:-1, 2:] + luma[2:, 2:] - luma[:-2, :-2] - 2 * luma[1:-1, :-2] - luma[2:, :-2]) / 8
    gy = (luma[2:, :-2] + 2 * luma[2:, 1:-1] + luma[2:, 2:] - luma[:-2, :-2] - 2 * luma[:-2, 1:-1] - luma[:-2, 2:]) / 8
    return float((np.hypot(gx, gy) > threshold).mean())


def dominant_color(rgb: np.ndarray) -> str:
    """Nome da cor mais frequente (categorica), pixel a pixel no espaco HSV."""
    hsv = np.asarray(Image.fromarray(rgb).convert("HSV"), dtype=np.float64)
    hue, sat, val = hsv[..., 0] * 360 / 255, hsv[..., 1] / 255, hsv[..., 2] / 255
    labels = np.empty(hue.shape, dtype=np.int64)
    index = {name: i for i, name in enumerate(COLOR_NAMES)}
    achromatic = (sat < 0.2) | (val < 0.2)
    labels[achromatic] = index["gray"]
    labels[achromatic & (val < 0.2)] = index["black"]
    labels[achromatic & (val >= 0.2) & (val > 0.8)] = index["white"]
    lower = 0.0
    for name, upper in HUE_NAMES:
        labels[~achromatic & (hue >= lower) & (hue < upper)] = index[name]
        lower = upper
    return COLOR_NAMES[int(np.bincount(labels.ravel(), minlength=len(COLOR_NAMES)).argmax())]


def extract(path: Path) -> dict[str, object]:
    rgb = np.asarray(Image.open(path).convert("RGB"))
    return {
        "image": path.name,
        "colorfulness": round(colorfulness(rgb), 2),
        "edge_density": round(edge_density(rgb), 4),
        "dominant_color": dominant_color(rgb),
    }


def main(paths: list[str]) -> None:
    for path in paths:
        print(extract(Path(path)))


if __name__ == "__main__":
    main(sys.argv[1:])
