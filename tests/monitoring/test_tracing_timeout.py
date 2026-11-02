"""O BentoML cancela a requisicao que passa de `traffic.timeout` (504 emitido por um middleware mais externo).

A linha `response` tem de existir mesmo assim: a funcao `/predict` comeca a rodar (linha `request`) e,
sem ela, o log ficaria com uma requisicao sem fim.
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import Iterator
from pathlib import Path

import pytest

from luminus_mlops.monitoring.logging_setup import LOG_FILE_NAME, LOGGER_NAME
from luminus_mlops.monitoring.tracing import Message, Receive, Scope, Send, TraceMiddleware


def _reset_serving_logger() -> None:
    """Tira os handlers (o do arquivo e os que o pytest/outros testes deixaram) para o logger recriar o seu."""
    logger = logging.getLogger(LOGGER_NAME)
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()


@pytest.fixture()
def log_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    monkeypatch.setenv("SERVING_LOG_DIR", str(tmp_path))
    yield tmp_path / LOG_FILE_NAME
    _reset_serving_logger()


def test_cancelled_request_still_gets_a_response_line(log_file: Path) -> None:
    _reset_serving_logger()
    async def slow_app(scope: Scope, receive: Receive, send: Send) -> None:
        await asyncio.sleep(60)

    async def sink(message: Message) -> None:
        return None

    async def run() -> None:
        async def receive() -> Message:
            return {"type": "http.request", "body": b"", "more_body": False}

        scope: Scope = {"type": "http", "path": "/predict", "method": "POST", "client": ("127.0.0.1", 1)}
        task = asyncio.ensure_future(TraceMiddleware(slow_app)(scope, receive, sink))
        await asyncio.sleep(0.05)
        task.cancel()  # o que o TimeoutMiddleware do BentoML faz aos 60 s
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(run())
    events = [json.loads(line) for line in log_file.read_text(encoding="utf-8").splitlines()]
    assert [e["event"] for e in events] == ["response"]
    assert events[0]["status_code"] == 504 and events[0]["error"] == "TimeoutError/request_timeout"
    assert events[0]["level"] == "ERROR"
