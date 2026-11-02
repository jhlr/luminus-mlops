"""Privacidade (SDD 5.4) com uma falha 5xx injetada pelo teste, nunca pelo codigo de producao.

Roda o servico no mesmo processo (TestClient): a falha vem de um monkeypatch de `ImageClassifier`
feito aqui. O log nao pode ter texto de excecao nem IP cru do cliente, e o campo de erro tem
so a classe da excecao mais um codigo fixo.
"""

from __future__ import annotations

import io
import json
import logging
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pytest
from PIL import Image
from starlette.testclient import TestClient

from luminus_mlops.inference import ImageClassifier, Prediction
from luminus_mlops.monitoring.logging_setup import LOGGER_NAME
from luminus_mlops.service import ClassifierService

ERROR_TEXT = "secret-exception-text-4242"
CATEGORIES = tuple(f"class_{i}" for i in range(10))


def _reset_logger() -> None:
    logger = logging.getLogger(LOGGER_NAME)
    for handler in list(logger.handlers):
        handler.close()
        logger.removeHandler(handler)


@pytest.fixture()
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[TestClient]:
    monkeypatch.setenv("SERVING_LOG_DIR", str(tmp_path))
    _reset_logger()
    with TestClient(ClassifierService.to_asgi(), raise_server_exceptions=False) as test_client:
        yield test_client
    _reset_logger()


def _png() -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (40, 30), (120, 60, 30)).save(buffer, "PNG")
    return buffer.getvalue()


def _fake_prediction(self: ImageClassifier, image: Image.Image) -> Prediction:
    probs = np.zeros(len(CATEGORIES))
    probs[3], probs[5] = 0.7, 0.3
    return Prediction(probs=probs, categories=CATEGORIES)


def test_log_has_no_exception_text_or_ip_even_on_4xx_and_5xx(client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    files = {"image": ("a.png", _png(), "image/png")}
    monkeypatch.setattr(ImageClassifier, "predict_image", _fake_prediction)
    ok = client.post("/predict", files=files)
    assert ok.status_code == 200 and ok.json()["label"] == "class_3"

    bad = client.post("/predict", data={"other": "x"})
    assert bad.status_code == 400

    def boom(self: ImageClassifier, image: Image.Image) -> Prediction:
        raise RuntimeError(ERROR_TEXT)

    monkeypatch.setattr(ImageClassifier, "predict_image", boom)
    broken = client.post("/predict", files=files)
    assert broken.status_code == 500

    text = (tmp_path / "inference.log").read_text(encoding="utf-8")
    events = [json.loads(line) for line in text.splitlines()]
    assert [(e["event"], e.get("status_code")) for e in events] == [
        ("request", None), ("response", 200), ("response", 400), ("request", None), ("response", 500)
    ]
    for secret in (ERROR_TEXT, "testclient", "127.0.0.1"):
        assert secret not in text
    assert events[-1]["error"] == "RuntimeError/inference_failed" and events[-1]["level"] == "ERROR"
    assert events[1]["out_predicted_class"] == "class_3" and events[1]["out_confidence"] == 0.7
    assert events[-1]["out_predicted_class"] is None
