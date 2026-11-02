"""Gera 5 imagens sinteticas de exemplo em `samples/` (seed fixa, sem baixar dataset).

Servem para o `just predict`, para os testes e para `load_gen.py --images-dir samples`. Nao sao
fotos reais: sao gradientes e ruido coloridos, so para exercitar o servico e o log.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

OUT_DIR = Path(__file__).resolve().parents[1] / "samples"
SIZE = (320, 240)
NAMES = ("sample_01.png", "sample_02.png", "sample_03.png", "sample_04.png", "sample_dark.png")


def _make(index: int, rng: np.random.Generator) -> Image.Image:
    width, height = SIZE
    x = np.linspace(0, 1, width)[None, :, None]
    y = np.linspace(0, 1, height)[:, None, None]
    base = rng.uniform(40, 220, size=(1, 1, 3))
    tilt = rng.uniform(-80, 80, size=(1, 1, 3))
    pixels = base + tilt * (x if index % 2 == 0 else y) + rng.normal(0, 10 + 6 * index, size=(height, width, 3))
    if index == 4:  # dark and blue-ish
        pixels = pixels * 0.25
        pixels[..., 2] *= 1.6
    return Image.fromarray(np.clip(pixels, 0, 255).astype(np.uint8), "RGB")


def main() -> None:
    OUT_DIR.mkdir(exist_ok=True)
    rng = np.random.default_rng(0)
    for index, name in enumerate(NAMES):
        _make(index, rng).save(OUT_DIR / name)
    print(f"{len(NAMES)} imagens em {OUT_DIR}")


if __name__ == "__main__":
    main()
