"""Rastro de uma requisicao `/predict`: objeto mutavel + middleware ASGI (SDD secao 5.3).

O middleware gera o `request_id`, cronometra e grava a linha `response` quando a resposta esta pronta
(logo antes de enviar o ultimo trecho do corpo, para o log nunca ficar atras do que o cliente ve). A funcao da API
recebe o mesmo `RequestTrace` por `ctx.request.state.trace` (nao por ContextVar: o BentoML roda
APIs sincronas numa thread com contexto copiado), grava a linha `request` assim que a imagem
decodifica e preenche as variaveis de saida. Dono unico dos eventos do log.

Privacidade (Decisao 6): so numeros e booleanos derivados; `error` e a classe da excecao mais um
codigo fixo, nunca a mensagem; o IP so entra como hash.
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from collections.abc import Awaitable, Callable, Iterator, MutableMapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

from starlette.datastructures import MutableHeaders

from .features import FEATURE_COLUMNS, ImageFeatures
from .logging_setup import hash_client_ip, service_version, write_event
from .outputs import OUTPUT_COLUMNS, OutputVariables

REQUEST_ID_HEADER = "X-Request-ID"
VALIDATION_ERROR = "ValidationError/invalid_request"
TIMEOUT_STATUS = 504
TIMEOUT_ERROR = "TimeoutError/request_timeout"

Scope = MutableMapping[str, Any]
Message = MutableMapping[str, Any]
Receive = Callable[[], Awaitable[Message]]
Send = Callable[[Message], Awaitable[None]]
ASGIApp = Callable[[Scope, Receive, Send], Awaitable[None]]


def http_error_code(status: int) -> str:
    """Erro de requisicao que nao chegou a funcao da API (so o status e conhecido)."""
    return VALIDATION_ERROR if status in (400, 422) else f"HTTPError/http_{status}"


@dataclass
class RequestTrace:
    request_id: str
    started: float
    endpoint: str
    method: str
    client_ip: str | None
    version: str
    outputs: OutputVariables | None = None
    error: str | None = None
    response_logged: bool = False
    extras: dict[str, object] = field(default_factory=dict)

    def _base(self, event: str) -> dict[str, object]:
        return {"event": event, "request_id": self.request_id, "endpoint": self.endpoint, "method": self.method}

    def log_request(self, features: ImageFeatures) -> None:
        """Linha `request`: variaveis da imagem prefixadas com `in_` (nunca os pixels)."""
        fields = self._base("request")
        fields["service_version"] = self.version
        fields["client_ip"] = self.client_ip
        for name, value in features.as_dict().items():
            fields[f"in_{name}"] = value
        write_event(logging.INFO, fields)

    def set_outputs(self, outputs: OutputVariables) -> None:
        self.outputs = outputs

    @contextmanager
    def stage(self, code: str) -> Iterator[None]:
        """Marca falhas da etapa com `Classe/codigo` (sem a mensagem) e deixa a excecao seguir."""
        try:
            yield
        except Exception as exc:
            self.error = f"{type(exc).__name__}/{code}"
            raise

    def log_response(self, status_code: int) -> None:
        if self.response_logged:
            return
        self.response_logged = True
        fields = self._base("response")
        fields["status_code"] = status_code
        fields["latency_ms"] = round((time.perf_counter() - self.started) * 1000, 3)
        out = self.outputs.as_dict() if self.outputs is not None and status_code < 400 else {}
        for name in OUTPUT_COLUMNS:
            fields[f"out_{name}"] = out.get(name)
        if status_code >= 400 and self.error is None:
            self.error = http_error_code(status_code)
        fields["error"] = self.error if status_code >= 400 else None
        write_event(logging.ERROR if status_code >= 500 else logging.INFO, fields)


class TraceMiddleware:
    """Middleware ASGI puro; so os `endpoints` listados geram eventos (`/metrics`, `/livez`... nao)."""

    def __init__(self, app: ASGIApp, endpoints: Sequence[str] = ("/predict",)) -> None:
        self.app = app
        self.endpoints = frozenset(endpoints)
        self.version = service_version()

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http" or scope["path"] not in self.endpoints:
            await self.app(scope, receive, send)
            return
        client = scope.get("client")
        trace = RequestTrace(
            request_id=str(uuid.uuid4()),
            started=time.perf_counter(),
            endpoint=scope["path"],
            method=scope["method"],
            client_ip=hash_client_ip(client[0] if client else None),
            version=self.version,
        )
        scope.setdefault("state", {})["trace"] = trace
        status = 500

        async def send_with_id(message: Message) -> None:
            nonlocal status
            if message["type"] == "http.response.start":
                status = int(message["status"])
                MutableHeaders(scope=message)[REQUEST_ID_HEADER] = trace.request_id
            elif message["type"] == "http.response.body" and not message.get("more_body", False):
                trace.log_response(status)  # a linha existe no arquivo antes de o cliente ver o fim da resposta
            await send(message)

        try:
            await self.app(scope, receive, send_with_id)
        except asyncio.CancelledError:
            # O TimeoutMiddleware do BentoML (mais externo) cancela a requisicao que passa de `traffic.timeout` e ele
            # mesmo envia o 504: a resposta nao passa por aqui, entao a linha `response` sai neste ponto.
            trace.error = TIMEOUT_ERROR
            trace.log_response(TIMEOUT_STATUS)
            raise
        except Exception as exc:
            if trace.error is None:
                trace.error = f"{type(exc).__name__}/unhandled_error"
            trace.log_response(500)
            raise
        trace.log_response(status)
