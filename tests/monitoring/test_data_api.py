"""Rotas `GET /data/*`: `/data/variables` (inferencia e monitor) e a analise do monitor."""

from __future__ import annotations

import asyncio
import csv
import io
from pathlib import Path

import httpx
import pytest
from starlette.applications import Starlette

from luminus_mlops.monitoring.analysis import DriftMonitor, compute_drift, load_reference, window_rows
from luminus_mlops.monitoring.data_api import build_data_app, hist_csv_columns
from luminus_mlops.monitoring.drift import Level
from luminus_mlops.monitoring.log_reader import LogReader
from luminus_mlops.monitoring.logging_setup import LOG_FILE_NAME
from luminus_mlops.monitoring.split import N_REFERENCE

from .test_log_reader import _append, _request, _response

ARTIFACTS = Path(__file__).resolve().parents[2] / "artifacts"


class _Client:
    """Cliente sincrono sobre o app ASGI (httpx + ASGITransport, sem subir porta)."""

    def __init__(self, app: Starlette) -> None:
        self.app = app

    def request(self, method: str, path: str, params: dict[str, str] | None = None) -> httpx.Response:
        async def call() -> httpx.Response:
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=self.app), base_url="http://data") as client:
                return await client.request(method, path, params=params)

        return asyncio.run(call())

    def get(self, path: str, params: dict[str, str] | None = None) -> httpx.Response:
        return self.request("GET", path, params)

    def post(self, path: str) -> httpx.Response:
        return self.request("POST", path)


def _client(log_dir: Path) -> _Client:
    return _Client(build_data_app(lambda: LogReader(log_dir)))


def _fill(log_dir: Path, ids: list[str]) -> None:
    log = log_dir / LOG_FILE_NAME
    for i, rid in enumerate(ids):
        _append(log, _request(rid, f"2026-11-01T10:00:{i:02d}.000001+00:00"), _response(rid, f"2026-11-01T10:00:{i:02d}.500000+00:00"))
    _append(log, _response("orphan", "2026-11-01T10:01:00.000000+00:00", 400))


def test_variables_json_csv_and_hist_columns(tmp_path: Path) -> None:
    _fill(tmp_path, ["a", "b", "c"])
    client = _client(tmp_path)
    rows = client.get("/variables").json()
    assert [r["request_id"] for r in rows] == ["a", "b", "c", "orphan"]
    assert "in_hist_l" not in rows[0]
    assert len(client.get("/variables", params={"include_hist": "true"}).json()[0]["in_hist_l"]) == 256

    table = list(csv.reader(io.StringIO(client.get("/variables", params={"format": "csv"}).text)))
    assert len(table) - 1 == 4 and "in_hist_r_000" not in table[0]
    wide = list(csv.reader(io.StringIO(client.get("/variables", params={"format": "csv", "include_hist": "true"}).text)))
    assert set(hist_csv_columns()) <= set(wide[0]) and len(wide[0]) == len(table[0]) + 1024
    assert wide[1][wide[0].index("in_hist_l_255")] == "16"
    assert wide[4][wide[0].index("in_width")] == ""  # linha so com resposta: entradas nulas


def test_variables_filters_and_rejects_bad_queries(tmp_path: Path) -> None:
    _fill(tmp_path, ["a", "b", "c"])
    client = _client(tmp_path)
    assert [r["request_id"] for r in client.get("/variables", params={"limit": "2"}).json()] == ["c", "orphan"]
    assert [r["request_id"] for r in client.get("/variables", params={"since": "2026-11-01T10:00:01+00:00", "until": "2026-11-01T10:00:01.9+00:00"}).json()] == ["b"]
    for params in ({"limit": "0"}, {"limit": "100001"}, {"format": "xml"}, {"since": "ontem"}, {"include_hist": "talvez"}):
        assert client.get("/variables", params=params).status_code == 422, params


def test_variables_is_get_only(tmp_path: Path) -> None:
    assert _client(tmp_path).post("/variables").status_code == 405


@pytest.mark.skipif(not (ARTIFACTS / "reference.csv").exists(), reason="artifacts/reference.csv ausente (just build-reference)")
def test_drift_report_levels_and_psi_cover_categoricals(tmp_path: Path) -> None:
    reference = load_reference(ARTIFACTS)
    assert reference.n_images == N_REFERENCE
    reader = LogReader(tmp_path)
    monitor = DriftMonitor(reader, reference, window_images=100, refresh_seconds=15.0)
    assert monitor.current().worst is Level.INCONCLUSIVE and monitor.current().n_window == 0  # sem log: inconclusivo

    log = tmp_path / LOG_FILE_NAME
    for i in range(10):
        _append(log, _request(f"r{i}", f"2026-11-01T10:00:{i:02d}.000001+00:00", mean=50.0 + i), _response(f"r{i}", f"2026-11-01T10:00:{i:02d}.500000+00:00"))
    reader.refresh()
    report = compute_drift(reference, window_rows(reader, None, None, 100))
    assert report.n_window == 10 and set(report.channels) == {"L", "R", "G", "B"}
    assert all(c.d is not None and c.level is not Level.INCONCLUSIVE for c in report.channels.values())
    assert {"predicted_class", "luminance_mean", "sharpness", "confidence", "entropy"} <= set(report.psi)
    # o calculo e reaproveitado dentro dos 15 s; /data/drift com since/until recalcula na hora
    first = monitor.current()
    _append(log, _request("late", "2026-11-01T10:05:00.000001+00:00"), _response("late", "2026-11-01T10:05:00.500000+00:00"))
    assert monitor.current() is first
    assert monitor.report("2026-11-01T10:00:00+00:00", None).n_window == 11
