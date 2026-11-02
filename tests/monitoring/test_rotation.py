"""Rotacao contra um servico em execucao (Decisao 4): SERVING_LOG_MAX_BYTES baixo gera `inference.log.1`."""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

import pytest

from .conftest import LiveService, start_service, stop_service


@pytest.fixture()
def tiny_service(tmp_path: Path) -> Iterator[LiveService]:
    live = start_service(tmp_path / "serving-logs", {"SERVING_LOG_MAX_BYTES": "20000"})
    yield live
    stop_service(live)


def test_rotation_in_running_service(tiny_service: LiveService) -> None:
    for _ in range(8):
        assert tiny_service.predict("sample_01.png").status_code == 200
    rotated = tiny_service.log_dir / "inference.log.1"
    assert rotated.exists()
    total = 0
    for path in sorted(tiny_service.log_dir.glob("inference.log*")):
        for line in path.read_text(encoding="utf-8").splitlines():
            json.loads(line)
            total += 1
    assert total == 16  # 8 requisicoes x (request + response): nada perdido na rotacao
