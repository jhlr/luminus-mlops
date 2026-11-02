from __future__ import annotations

import json
from pathlib import Path

from luminus_mlops.monitoring.features import FEATURE_COLUMNS, HIST_FEATURES
from luminus_mlops.monitoring.log_reader import INPUT_HIST_COLUMNS, LogReader
from luminus_mlops.monitoring.logging_setup import LOG_FILE_NAME


def _request(rid: str, ts: str, mean: float = 100.0) -> dict[str, object]:
    event: dict[str, object] = {"ts": ts, "level": "INFO", "event": "request", "request_id": rid,
                                "endpoint": "/predict", "method": "POST", "in_luminance_mean": mean, "in_width": 4, "in_height": 4}
    for name in HIST_FEATURES:
        event[f"in_{name}"] = [0] * 255 + [16]
    return event


def _response(rid: str, ts: str, status: int = 200) -> dict[str, object]:
    return {"ts": ts, "level": "INFO", "event": "response", "request_id": rid, "endpoint": "/predict", "method": "POST",
            "status_code": status, "latency_ms": 12.5, "out_predicted_class": "tench", "out_confidence": 0.9, "error": None}


def _append(path: Path, *events: dict[str, object]) -> int:
    data = "".join(json.dumps(e) + "\n" for e in events).encode()
    with path.open("ab") as fh:
        fh.write(data)
    return len(data)


def test_join_by_request_id_with_orphan_response(tmp_path: Path) -> None:
    log = tmp_path / LOG_FILE_NAME
    _append(log, _request("a", "2026-11-01T10:00:00.000001+00:00"), _response("a", "2026-11-01T10:00:00.500000+00:00"),
            _response("b", "2026-11-01T10:00:01.000000+00:00", 400))
    reader = LogReader(tmp_path)
    assert reader.refresh() == 3
    rows = reader.rows()
    assert [r["request_id"] for r in rows] == ["a", "b"]
    assert rows[0]["in_luminance_mean"] == 100.0 and rows[0]["out_predicted_class"] == "tench" and str(rows[0]["ts"]).endswith("000001+00:00")
    assert rows[1]["in_luminance_mean"] is None and rows[1]["status_code"] == 400
    assert all(c not in rows[0] for c in INPUT_HIST_COLUMNS)
    with_hist = reader.rows(include_hist=True)
    assert with_hist[0]["in_hist_l"] == [0] * 255 + [16] and with_hist[1]["in_hist_l"] is None
    assert set(f"in_{c}" for c in FEATURE_COLUMNS if c not in HIST_FEATURES) <= set(rows[0])


def test_incremental_refresh_reads_only_new_bytes(tmp_path: Path) -> None:
    log = tmp_path / LOG_FILE_NAME
    first = _append(log, _request("a", "2026-11-01T10:00:00.000000+00:00"), _response("a", "2026-11-01T10:00:01.000000+00:00"))
    reader = LogReader(tmp_path)
    reader.refresh()
    assert reader.stats.bytes_read == first
    assert list(reader.offsets.values()) == [first]
    assert reader.refresh() == 0 and reader.stats.bytes_read == first  # nada novo: nao rele
    second = _append(log, _request("b", "2026-11-01T10:00:02.000000+00:00"), _response("b", "2026-11-01T10:00:03.000000+00:00"))
    assert reader.refresh() == 2
    assert reader.stats.bytes_read == first + second
    assert list(reader.offsets.values()) == [first + second]
    assert len(reader.rows()) == 2


def test_partial_last_line_is_kept_for_the_next_refresh(tmp_path: Path) -> None:
    log = tmp_path / LOG_FILE_NAME
    line = json.dumps(_response("a", "2026-11-01T10:00:00.000000+00:00")) + "\n"
    log.write_bytes(line.encode() + line.encode()[:20])
    reader = LogReader(tmp_path)
    assert reader.refresh() == 1
    with log.open("ab") as fh:
        fh.write(line.encode()[20:])
    assert reader.refresh() == 1 and reader.stats.bad_lines == 0


def test_rotation_keeps_offsets_per_inode_and_joins_across_files(tmp_path: Path) -> None:
    log = tmp_path / LOG_FILE_NAME
    _append(log, _request("a", "2026-11-01T10:00:00.000000+00:00"))
    reader = LogReader(tmp_path)
    reader.refresh()
    log.rename(tmp_path / f"{LOG_FILE_NAME}.1")  # rotacao: o inode antigo vira .1
    _append(tmp_path / f"{LOG_FILE_NAME}.1", _request("b", "2026-11-01T10:00:00.100000+00:00"))
    _append(log, _response("a", "2026-11-01T10:00:01.000000+00:00"), _response("b", "2026-11-01T10:00:02.000000+00:00"))
    assert reader.refresh() == 3  # a linha nova do .1 e as 2 do arquivo novo; nada relido
    assert [r["request_id"] for r in reader.rows()] == ["a", "b"]
    assert reader.rows()[0]["in_luminance_mean"] == 100.0


def test_window_filters_limit_and_bounded_deque(tmp_path: Path) -> None:
    log = tmp_path / LOG_FILE_NAME
    _append(log, *[_response(str(i), f"2026-11-01T10:00:0{i}.000000+00:00") for i in range(6)])
    reader = LogReader(tmp_path, max_rows=4)
    reader.refresh()
    assert [r["request_id"] for r in reader.rows()] == ["2", "3", "4", "5"]  # so os 4 mais novos ficam na memoria
    assert [r["request_id"] for r in reader.rows(limit=2)] == ["4", "5"]
    assert [r["request_id"] for r in reader.rows(since="2026-11-01T10:00:03+00:00", until="2026-11-01T07:00:04.5-03:00")] == ["3", "4"]
