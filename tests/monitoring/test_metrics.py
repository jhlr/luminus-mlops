"""`/metrics` em formato Prometheus valido, com as series do BentoML e as proprias (D2)."""

from __future__ import annotations

import httpx
from prometheus_client.parser import text_string_to_metric_families

from luminus_mlops.monitoring import metrics as m

from .conftest import LiveService


def _samples(text: str) -> dict[str, list[tuple[dict[str, str], float]]]:
    out: dict[str, list[tuple[dict[str, str], float]]] = {}
    for family in text_string_to_metric_families(text):
        for sample in family.samples:
            out.setdefault(sample.name, []).append((sample.labels, sample.value))
    return out


def test_metrics_parse_and_contain_expected_series(service: LiveService) -> None:
    service.predict("sample_01.png")
    service.predict("sample_dark.png")
    response = httpx.get(f"{service.url}/metrics", timeout=30)
    assert response.status_code == 200
    samples = _samples(response.text)  # levanta ValueError se o texto nao for exposicao Prometheus valida

    assert "bentoml_service_request_total" in samples
    assert any(s[0].get("endpoint") == "/predict" for s in samples["bentoml_service_request_total"])
    buckets = [labels["le"] for labels, _ in samples["bentoml_service_request_duration_seconds_bucket"] if labels.get("endpoint") == "/predict"]
    assert [float(b) for b in buckets if b != "+Inf"] == list(m.DURATION_BUCKETS)
    assert "luminus_input_luminance_mean_bucket" in samples
    assert "luminus_prediction_confidence_bucket" in samples
    assert "luminus_prediction_entropy_bucket" in samples
    bands = {labels["band"]: value for labels, value in samples["luminus_prediction_confidence_band_total"]}
    assert set(bands) == {"low", "medium", "high"}
    assert sum(bands.values()) >= 2
