"""Variaveis de saida do modelo (dono unico, SDD secao 5.2) e a regra do top-k.

`top_k` e a MESMA funcao que o servico usa para montar a resposta e que este modulo usa para o
log: nao existe segunda copia da regra. So numeros derivados e o nome da classe prevista sao
gravados no log, nunca a imagem.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from ..inference import Prediction

TOP_K = 5
BAND_LOW = 0.5  # confianca abaixo disto: "low"
BAND_HIGH = 0.8  # confianca igual ou acima disto: "high"
CONFIDENCE_BANDS: tuple[str, ...] = ("low", "medium", "high")

NUMERIC_OUTPUTS: tuple[str, ...] = ("confidence", "margin", "entropy", "top5_mass")
CATEGORICAL_OUTPUTS: tuple[str, ...] = ("predicted_class",)
OUTPUT_COLUMNS: tuple[str, ...] = ("predicted_class", "confidence", "margin", "entropy", "top5_mass")
SCALAR_OUTPUT_COLUMNS: tuple[str, ...] = OUTPUT_COLUMNS


def top_k(pred: Prediction, k: int = TOP_K) -> list[tuple[str, float]]:
    """As `k` classes mais provaveis como (nome, probabilidade), da melhor para a pior."""
    order = np.argsort(-pred.probs, kind="stable")[:k]
    return [(pred.categories[int(i)], float(pred.probs[i])) for i in order]


def confidence_band(confidence: float) -> str:
    if confidence < BAND_LOW:
        return "low"
    if confidence < BAND_HIGH:
        return "medium"
    return "high"


@dataclass(frozen=True)
class OutputVariables:
    predicted_class: str
    confidence: float  # probabilidade softmax do top-1
    margin: float  # probabilidade do top-1 menos a do top-2
    entropy: float  # entropia de Shannon do softmax, em bits
    top5_mass: float  # probabilidade total das 5 melhores classes

    def as_dict(self) -> dict[str, object]:
        return {name: getattr(self, name) for name in OUTPUT_COLUMNS}

    def scalars(self) -> dict[str, object]:
        return self.as_dict()


def derive_outputs(pred: Prediction) -> OutputVariables:
    best = top_k(pred)
    probs = pred.probs[pred.probs > 0]
    entropy = float(-np.sum(probs * np.log2(probs)))
    return OutputVariables(
        predicted_class=best[0][0],
        confidence=round(best[0][1], 4),
        margin=round(best[0][1] - best[1][1], 4),
        entropy=round(entropy if not math.isnan(entropy) else 0.0, 4),
        top5_mass=round(sum(p for _, p in best), 4),
    )
