"""Response-contract guard: monitoring must not change `/predict`.

Fields and types captured from the service BEFORE the monitoring code was wired in. Values are not compared.
"""

from __future__ import annotations

import pytest

from .conftest import SAMPLE_NAMES, LiveService

CONTRACT: dict[str, type] = {"label": str, "confidence": float, "top5": list}


@pytest.mark.parametrize("name", SAMPLE_NAMES)
def test_predict_response_contract(service: LiveService, name: str) -> None:
    response = service.predict(name)
    assert response.status_code == 200
    body = response.json()
    assert set(body) == set(CONTRACT)
    for field, expected in CONTRACT.items():
        assert isinstance(body[field], expected), f"{field}: {type(body[field]).__name__}"
    assert len(body["top5"]) == 5
    for item in body["top5"]:
        assert set(item) == {"label", "confidence"} and isinstance(item["label"], str) and isinstance(item["confidence"], float)
    assert body["top5"][0] == {"label": body["label"], "confidence": body["confidence"]}
