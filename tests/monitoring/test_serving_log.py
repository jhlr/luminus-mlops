"""Guardas do log de servico (D1): contagem de linhas, ruido, privacidade, append-only, vazamento entre requisicoes."""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from typing import Any, cast
from pathlib import Path

import httpx

from .conftest import SAMPLE_NAMES, LiveService

TS_PATTERN_LEN = len("2026-11-01T14:03:22.481516+00:00")


def _event_lines(service: LiveService, request_id: str) -> list[dict[str, object]]:
    return [e for e in service.log_events() if e["request_id"] == request_id]


def test_one_valid_and_one_invalid_request_write_exactly_three_lines(fresh_service: LiveService) -> None:
    valid = fresh_service.predict("sample_03.png")
    assert valid.status_code == 200
    invalid = httpx.post(f"{fresh_service.url}/predict", data={"foo": "bar"}, timeout=30)
    assert invalid.status_code == 400

    lines = fresh_service.log_lines()
    assert len(lines) == 3
    events = [json.loads(line) for line in lines]
    for event in events:
        assert len(str(event["ts"])) == TS_PATTERN_LEN
        parsed = datetime.fromisoformat(str(event["ts"]))
        offset = parsed.utcoffset()
        assert offset is not None and offset.total_seconds() == 0

    request_id = valid.headers["x-request-id"]
    kinds = [(e["event"], e["request_id"]) for e in events]
    assert kinds.index(("request", request_id)) < kinds.index(("response", request_id))
    request = next(e for e in events if e["event"] == "request" and e["request_id"] == request_id)
    pixels = cast(int, request["in_width"]) * cast(int, request["in_height"])
    for channel in ("l", "r", "g", "b"):
        histogram = request[f"in_hist_{channel}"]
        assert isinstance(histogram, list) and sum(histogram) == pixels
    invalid_id = invalid.headers["x-request-id"]
    invalid_events = _event_lines(fresh_service, invalid_id)
    assert [e["event"] for e in invalid_events] == ["response"]
    assert invalid_events[0]["status_code"] == 400
    assert all(invalid_events[0][k] is None for k in invalid_events[0] if k.startswith("out_"))


def test_metrics_and_health_endpoints_write_no_log_lines(service: LiveService) -> None:
    service.predict("sample_01.png")
    before = len(service.log_lines())
    for path in ("/metrics", "/livez", "/readyz"):
        assert httpx.get(f"{service.url}{path}", timeout=30).status_code == 200
    assert len(service.log_lines()) == before


def test_log_is_append_only(service: LiveService) -> None:
    service.predict("sample_02.png")
    before = service.log_file.read_bytes()
    service.predict("sample_04.png")
    after = service.log_file.read_bytes()
    assert len(after) > len(before)
    assert after.startswith(before)


def test_concurrent_requests_do_not_cross_talk(service: LiveService) -> None:
    names = SAMPLE_NAMES * 3

    def call(name: str) -> tuple[str, dict[str, Any]]:
        response = service.predict(name)
        assert response.status_code == 200
        return response.headers["x-request-id"], response.json()

    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(call, names))
    responses = {e["request_id"]: e for e in service.log_events() if e["event"] == "response"}
    for request_id, body in results:
        logged = responses[request_id]
        assert logged["out_predicted_class"] == body["label"]
        assert logged["out_confidence"] == body["confidence"]


def test_log_never_contains_raw_ip_or_input_text(service: LiveService) -> None:
    assert service.predict("sample_03.png").status_code == 200
    httpx.post(f"{service.url}/predict", content=b"not-an-image-secret", headers={"content-type": "text/plain"}, timeout=30)
    httpx.post(f"{service.url}/predict", data={"foo": "bar"}, timeout=30)
    text = service.log_file.read_text(encoding="utf-8")
    for secret in ("127.0.0.1", "not-an-image-secret"):
        assert secret not in text


def test_every_line_is_a_valid_json_object(service: LiveService) -> None:
    service.predict("sample_02.png")
    for line in service.log_lines():
        assert isinstance(json.loads(line), dict)


def test_log_dir_is_the_configured_one(service: LiveService) -> None:
    assert Path(service.log_file).exists()


def test_data_variables_writes_no_log_lines_and_matches_the_log(service: LiveService) -> None:
    service.predict("sample_01.png")
    before = service.log_lines()
    for query in ("", "?include_hist=true", "?format=csv", "?format=csv&include_hist=true"):
        assert httpx.get(f"{service.url}/data/variables{query}", timeout=30).status_code == 200
    assert service.log_lines() == before
    distinct = {json.loads(line)["request_id"] for line in before}
    rows = httpx.get(f"{service.url}/data/variables?format=csv", timeout=30).text.splitlines()
    assert len(rows) - 1 == len(distinct)
