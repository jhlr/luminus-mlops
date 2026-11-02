from __future__ import annotations

import math

import numpy as np
import pytest

from luminus_mlops.inference import Prediction
from luminus_mlops.monitoring import outputs as o

CATEGORIES = tuple(f"class_{i}" for i in range(8))


def _pred(*probs: float) -> Prediction:
    padded = np.array([*probs, *([0.0] * (len(CATEGORIES) - len(probs)))], dtype=np.float64)
    return Prediction(probs=padded, categories=CATEGORIES)


def test_top_k_orders_by_probability() -> None:
    best = o.top_k(_pred(0.1, 0.5, 0.3, 0.1), k=3)
    assert [name for name, _ in best] == ["class_1", "class_2", "class_0"]
    assert best[0][1] == pytest.approx(0.5)


def test_derived_variables() -> None:
    out = o.derive_outputs(_pred(0.5, 0.25, 0.25))
    assert out.predicted_class == "class_0"
    assert out.confidence == pytest.approx(0.5) and out.margin == pytest.approx(0.25)
    assert out.entropy == pytest.approx(1.5) and out.top5_mass == pytest.approx(1.0)
    assert tuple(out.as_dict()) == o.OUTPUT_COLUMNS


def test_uniform_softmax_has_maximum_entropy() -> None:
    out = o.derive_outputs(_pred(*([1 / 8] * 8)))
    assert out.entropy == pytest.approx(math.log2(8)) and out.margin == pytest.approx(0.0)
    assert out.top5_mass == pytest.approx(5 / 8)


def test_confidence_bands() -> None:
    assert [o.confidence_band(c) for c in (0.1, 0.5, 0.79, 0.8, 1.0)] == ["low", "medium", "medium", "high", "high"]
