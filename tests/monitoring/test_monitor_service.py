"""O servico `monitor` de verdade (subprocesso): rotas `/data/*` e gauges de drift em `/metrics`."""

from __future__ import annotations

import csv
import io
from collections.abc import Iterator
from pathlib import Path

import httpx
import pytest
from prometheus_client.parser import text_string_to_metric_families

from luminus_mlops.monitoring.logging_setup import LOG_FILE_NAME
from luminus_mlops.monitoring.split import N_REFERENCE

from .conftest import LiveService, start_service, stop_service
from .test_log_reader import _append, _request, _response

ARTIFACTS = Path(__file__).resolve().parents[2] / "artifacts"
MONITOR_TARGET = "luminus_mlops.monitor_service:MonitorService"
pytestmark = pytest.mark.skipif(not (ARTIFACTS / "reference.csv").exists(), reason="artifacts/reference.csv ausente")


@pytest.fixture(scope="module")
def monitor(tmp_path_factory: pytest.TempPathFactory) -> Iterator[LiveService]:
    log_dir = tmp_path_factory.mktemp("monitor-logs")
    log = log_dir / LOG_FILE_NAME
    for i in range(12):
        rid = f"req-{i}"
        _append(log, _request(rid, f"2026-11-01T10:00:{i:02d}.000001+00:00", mean=40.0 + i), _response(rid, f"2026-11-01T10:00:{i:02d}.500000+00:00"))
    _append(log, _response("bad", "2026-11-01T10:01:00.000000+00:00", 400))
    live = start_service(log_dir, extra_env={"ARTIFACTS_DIR": str(ARTIFACTS), "DRIFT_WINDOW_IMAGES": "100"}, target=MONITOR_TARGET)
    yield live
    stop_service(live)


def test_variables_matches_the_log_and_the_monitor_never_writes_it(monitor: LiveService) -> None:
    before = monitor.log_file.read_bytes()
    text = httpx.get(f"{monitor.url}/data/variables?format=csv", timeout=30).text
    assert len(list(csv.reader(io.StringIO(text)))) - 1 == 13
    wide = httpx.get(f"{monitor.url}/data/variables?format=csv&include_hist=true", timeout=30).text.splitlines()[0]
    assert "in_hist_b_255" in wide
    assert monitor.log_file.read_bytes() == before


def test_summary_drift_and_reference_routes(monitor: LiveService) -> None:
    summary = httpx.get(f"{monitor.url}/data/summary", timeout=30).json()
    assert summary["window_images"] == 12 and len(summary["histograms"]["L"]["window"]) == 256
    assert summary["numeric"]["luminance_mean"]["window"]["median"] == pytest.approx(45.5)
    assert summary["categorical"]["predicted_class"]["window"] == {"tench": 12}
    drift = httpx.get(f"{monitor.url}/data/drift?channel=all", timeout=30).json()
    assert set(drift["channels"]) == {"L", "R", "G", "B"} and drift["worst_level"] in {"green", "yellow", "red"}
    assert httpx.get(f"{monitor.url}/data/drift?channel=L&since=2026-11-01T10:00:00%2B00:00&until=2026-11-01T10:00:02.9%2B00:00", timeout=30).json()["window_images"] == 3
    ref = httpx.get(f"{monitor.url}/data/reference", timeout=30).json()
    assert len(ref) == N_REFERENCE and "ts" not in ref[0] and "request_id" not in ref[0]
    assert len(httpx.get(f"{monitor.url}/data/reference/hist", timeout=30).json()) == 256


def test_metrics_expose_drift_gauges_including_categorical_psi(monitor: LiveService) -> None:
    families = {f.name: f for f in text_string_to_metric_families(httpx.get(f"{monitor.url}/metrics", timeout=30).text)}
    for name in ("luminus_drift_ks_d", "luminus_drift_ks_pvalue", "luminus_drift_level", "luminus_drift_psi", "luminus_drift_window_images"):
        assert name in families, name
    psi_variables = {s.labels["variable"] for s in families["luminus_drift_psi"].samples}
    assert {"predicted_class", "luminance_mean", "sharpness", "confidence", "entropy"} <= psi_variables
    assert {s.labels["channel"] for s in families["luminus_drift_level"].samples} == {"L", "R", "G", "B"}
    assert families["luminus_drift_window_images"].samples[0].value == 12
