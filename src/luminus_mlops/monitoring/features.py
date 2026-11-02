"""Variaveis extraidas da imagem de entrada (dono unico).

Por que so estas: dao histogramas, boxplots, KS e PSI suficientes para ver mudanca de iluminacao, cor,
nitidez e tamanho da foto, sem guardar pixels. As medias saem dos histogramas 0-255
(`Image.histogram()`), como no codigo do curso. Adicionar ou remover uma variavel e editar so este arquivo.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from PIL import Image

HIST_BINS = 256
SCALAR_FEATURES: tuple[str, ...] = (
    "width",
    "height",
    "file_bytes",
    "luminance_mean",
    "r_mean",
    "g_mean",
    "b_mean",
    "luminance_std",
    "sharpness",
)
CATEGORICAL_FEATURES: tuple[str, ...] = ()  # nenhuma variavel categorica de entrada (as categoricas sao da saida)
HIST_FEATURES: tuple[str, ...] = ("hist_l", "hist_r", "hist_g", "hist_b")
# Ordem de colunas do log (sem prefixo): escalares e depois os histogramas.
FEATURE_COLUMNS: tuple[str, ...] = SCALAR_FEATURES + CATEGORICAL_FEATURES + HIST_FEATURES
CHANNELS: tuple[str, ...] = ("l", "r", "g", "b")


@dataclass(frozen=True)
class ImageFeatures:
    width: int
    height: int
    file_bytes: int | None
    luminance_mean: float
    r_mean: float
    g_mean: float
    b_mean: float
    luminance_std: float  # contraste
    sharpness: float  # variancia do Laplaciano: foto borrada tem valor baixo
    hist_l: tuple[int, ...]
    hist_r: tuple[int, ...]
    hist_g: tuple[int, ...]
    hist_b: tuple[int, ...]

    def as_dict(self) -> dict[str, object]:
        """Todas as variaveis, na ordem de `FEATURE_COLUMNS` (histogramas como listas)."""
        out: dict[str, object] = {name: getattr(self, name) for name in FEATURE_COLUMNS}
        for name in HIST_FEATURES:
            out[name] = list(getattr(self, name))
        return out

    def scalars(self) -> dict[str, object]:
        """So as variaveis escalares (colunas do reference.csv)."""
        return {name: getattr(self, name) for name in SCALAR_FEATURES}


def mean_from_histogram(counts: Sequence[int]) -> float:
    """mean = sum(i * counts[i]) / sum(counts), como no codigo do curso."""
    total = sum(counts)
    return sum(i * c for i, c in enumerate(counts)) / total if total else 0.0


def std_from_histogram(counts: Sequence[int]) -> float:
    total = sum(counts)
    if not total:
        return 0.0
    mean = mean_from_histogram(counts)
    return math.sqrt(sum(c * (i - mean) ** 2 for i, c in enumerate(counts)) / total)


def laplacian_variance(luma: np.ndarray) -> float:
    """Variancia do Laplaciano (kernel 3x3 de 4 vizinhos) do canal L."""
    if luma.shape[0] < 3 or luma.shape[1] < 3:
        return 0.0
    gray = luma.astype(np.int32)
    lap = gray[:-2, 1:-1] + gray[2:, 1:-1] + gray[1:-1, :-2] + gray[1:-1, 2:] - 4 * gray[1:-1, 1:-1]
    return float(lap.var(dtype=np.float64))


def uploaded_size(image: Image.Image) -> int | None:
    """Tamanho em bytes do arquivo enviado, lido do arquivo aberto pelo Pillow.

    Chamar ANTES de decodificar a imagem (o Pillow fecha o arquivo depois do `load`).
    """
    fp = getattr(image, "fp", None)
    if fp is None or not hasattr(fp, "seek") or not hasattr(fp, "tell"):
        return None
    position = fp.tell()
    fp.seek(0, 2)
    size = int(fp.tell())
    fp.seek(position)
    return size


def extract_features(image: Image.Image, file_bytes: int | None = None) -> ImageFeatures:
    """Calcula as variaveis da imagem; `file_bytes` e o tamanho do upload (`uploaded_size`)."""
    rgb = image.convert("RGB")
    luma_image = rgb.convert("L")
    raw = rgb.histogram()
    hist_r, hist_g, hist_b = raw[:HIST_BINS], raw[HIST_BINS : 2 * HIST_BINS], raw[2 * HIST_BINS :]
    hist_l = luma_image.histogram()
    return ImageFeatures(
        width=rgb.width,
        height=rgb.height,
        file_bytes=file_bytes,
        luminance_mean=mean_from_histogram(hist_l),
        r_mean=mean_from_histogram(hist_r),
        g_mean=mean_from_histogram(hist_g),
        b_mean=mean_from_histogram(hist_b),
        luminance_std=std_from_histogram(hist_l),
        sharpness=laplacian_variance(np.asarray(luma_image)),
        hist_l=tuple(hist_l),
        hist_r=tuple(hist_r),
        hist_g=tuple(hist_g),
        hist_b=tuple(hist_b),
    )
