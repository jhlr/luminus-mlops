"""Servico BentoML `monitor` (ambiente de MONITORAMENTO, SDD secao 7, E2).

Segundo servico, pequeno (sem torch/onnxruntime/pesos): le o log de servico e as referencias
SOMENTE LEITURA e expoe

  GET /data/variables   mesmas linhas da inferencia (via `log_reader`)
  GET /data/reference, /data/reference/hist, /data/summary, /data/drift
  GET /metrics          nativo do BentoML + gauges `luminus_drift_*` (janela deslizante)

Os gauges sao recalculados no maximo a cada 15 s: um middleware ASGI atualiza-os quando o
Prometheus raspa `/metrics` (sem thread de fundo). O monitor nunca escreve no diretorio de log.

Rodar:
  SERVING_LOG_DIR=./serving-logs bentoml serve luminus_mlops.monitor_service:MonitorService --port 3002
"""

from __future__ import annotations

import json
import os
import threading
from collections.abc import Mapping
from pathlib import Path

import bentoml
from starlette.requests import Request
from starlette.responses import Response
from starlette.routing import Route

from .monitoring.analysis import (
    CHANNEL_LABELS,
    DEFAULT_WINDOW_IMAGES,
    DriftMonitor,
    ReferenceMissing,
    drift_to_dict,
    load_reference,
)
from .monitoring.data_api import BadQuery, bad_query_response, build_data_app, get_log_reader, parse_format, parse_window, table_response
from .monitoring.metrics import get_drift_gauges
from .monitoring.tracing import ASGIApp, Receive, Scope, Send
from .paths import ROOT

__all__ = ["MonitorService"]

_monitor: DriftMonitor | None = None
_monitor_lock = threading.Lock()


def artifacts_dir() -> Path:
    return Path(os.environ.get("ARTIFACTS_DIR", ROOT / "artifacts"))


def window_images() -> int:
    return int(os.environ.get("DRIFT_WINDOW_IMAGES", DEFAULT_WINDOW_IMAGES))


def get_monitor() -> DriftMonitor:
    """Um `DriftMonitor` por processo (lazy, dentro do worker)."""
    global _monitor
    with _monitor_lock:
        if _monitor is None:
            _monitor = DriftMonitor(get_log_reader(), load_reference(artifacts_dir()), window_images())
        return _monitor


def _typed(cell: str) -> object:
    """Celula de CSV -> valor JSON (vazio = null, True/False = booleano, numeros = numeros)."""
    if cell == "":
        return None
    if cell in ("True", "False"):
        return cell == "True"
    for cast in (int, float):
        try:
            return cast(cell)
        except ValueError:
            continue
    return cell


def _reference_response(request: Request, table: list[dict[str, str]]) -> Response:
    try:
        fmt = parse_format(request)
    except BadQuery as exc:
        return bad_query_response(exc)
    columns = list(table[0]) if table else []
    rows: list[Mapping[str, object]] = [{key: _typed(value) for key, value in row.items()} for row in table]
    return table_response(rows, columns, fmt)


def reference_endpoint(request: Request) -> Response:
    return _reference_response(request, get_monitor().reference.table)


def reference_hist_endpoint(request: Request) -> Response:
    return _reference_response(request, get_monitor().reference.hist_table)


def summary_endpoint(request: Request) -> Response:
    try:
        since, until = parse_window(request)
    except BadQuery as exc:
        return bad_query_response(exc)
    return _json(get_monitor().summary(since, until))


def drift_endpoint(request: Request) -> Response:
    channel = request.query_params.get("channel", "all")
    if channel != "all" and channel not in CHANNEL_LABELS:
        return bad_query_response(BadQuery(f"channel deve ser all ou um de {tuple(CHANNEL_LABELS)}"))
    try:
        since, until = parse_window(request)
    except BadQuery as exc:
        return bad_query_response(exc)
    return _json(drift_to_dict(get_monitor().report(since, until), channel))


def _json(payload: object) -> Response:
    return Response(json.dumps(payload), media_type="application/json")


class DriftGaugeMiddleware:
    """Antes de `/metrics` responder, atualiza os gauges de drift (reaproveita o calculo de menos de 15 s)."""

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] == "http" and scope["path"] == "/metrics":
            get_drift_gauges().update(get_monitor().current())
        await self.app(scope, receive, send)


data_app = build_data_app(
    get_log_reader,
    extra_routes=[
        Route("/reference/hist", reference_hist_endpoint, methods=["GET"]),
        Route("/reference", reference_endpoint, methods=["GET"]),
        Route("/summary", summary_endpoint, methods=["GET"]),
        Route("/drift", drift_endpoint, methods=["GET"]),
    ],
)


@bentoml.asgi_app(data_app, path="/data")
class _Monitor:
    def __init__(self) -> None:
        try:
            self.monitor = get_monitor()  # falha cedo, com mensagem, se a referencia nao existe
        except ReferenceMissing as exc:
            raise SystemExit(str(exc)) from exc
        get_drift_gauges()


MonitorService = bentoml.service(name="monitor", workers=1)(_Monitor)
MonitorService.add_asgi_middleware(DriftGaugeMiddleware)
