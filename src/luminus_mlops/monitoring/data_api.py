"""Rotas `GET /data/*` montadas como app ASGI (Starlette) nos dois servicos (SDD secao 7, E2).

O BentoML 1.4 so serve POST em `@bentoml.api` (`_bentoml_impl/server/app.py` registra
`methods=["POST"]`), entao as rotas de dados entram pelo `@bentoml.asgi_app`, o fallback que o SDD
preve. Este modulo e o dono de `/data/variables` (mesmo comportamento na inferencia e no
`monitor`) e dos helpers de resposta JSON/CSV; so biblioteca padrao mais Starlette (que ja vem
com o BentoML), nada de pandas.
"""

from __future__ import annotations

import csv
import io
import json
import threading
from collections.abc import Callable, Mapping, Sequence

from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.routing import Route

from .log_reader import INPUT_HIST_COLUMNS, ROW_SCALAR_COLUMNS, LogReader, normalize_ts
from .logging_setup import log_dir

DEFAULT_LIMIT = 10_000
MAX_LIMIT = 100_000
FORMATS = ("json", "csv")
LIST_SEPARATOR = "|"

_reader: LogReader | None = None
_reader_lock = threading.Lock()


def get_log_reader() -> LogReader:
    """Um `LogReader` por processo (lazy, dentro do worker), sobre o `SERVING_LOG_DIR`."""
    global _reader
    with _reader_lock:
        if _reader is None:
            _reader = LogReader(log_dir())
        return _reader


class BadQuery(ValueError):
    """Parametro de consulta invalido (vira HTTP 422)."""


def parse_bool(value: str | None, name: str) -> bool:
    if value is None:
        return False
    lowered = value.strip().lower()
    if lowered in ("true", "1"):
        return True
    if lowered in ("false", "0", ""):
        return False
    raise BadQuery(f"{name} deve ser true ou false")


def parse_format(request: Request) -> str:
    fmt = request.query_params.get("format", "json")
    if fmt not in FORMATS:
        raise BadQuery(f"format deve ser um de {FORMATS}")
    return fmt


def parse_limit(request: Request) -> int:
    raw = request.query_params.get("limit")
    if raw is None:
        return DEFAULT_LIMIT
    if not raw.isdigit() or not 1 <= int(raw) <= MAX_LIMIT:
        raise BadQuery(f"limit deve ser um inteiro entre 1 e {MAX_LIMIT}")
    return int(raw)


def parse_window(request: Request) -> tuple[str | None, str | None]:
    """`since`/`until` ISO 8601 validados e normalizados (UTC, largura fixa)."""
    bounds: list[str | None] = []
    for name in ("since", "until"):
        raw = request.query_params.get(name)
        try:
            bounds.append(normalize_ts(raw))
        except ValueError as exc:
            raise BadQuery(f"{name} deve ser ISO 8601") from exc
    return bounds[0], bounds[1]


def hist_csv_columns() -> list[str]:
    """`in_hist_r_000..255` etc. (formato largo do CSV)."""
    return [f"{column}_{i:03d}" for column in INPUT_HIST_COLUMNS for i in range(256)]


def _cell(value: object) -> object:
    if isinstance(value, list):
        return LIST_SEPARATOR.join(str(item) for item in value)
    return "" if value is None else value


def rows_to_csv(rows: Sequence[Mapping[str, object]], columns: Sequence[str]) -> str:
    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(columns)
    for row in rows:
        writer.writerow([_cell(row.get(column)) for column in columns])
    return buffer.getvalue()


def table_response(rows: Sequence[Mapping[str, object]], columns: Sequence[str], fmt: str) -> Response:
    """JSON (lista de objetos) ou CSV (uma linha por objeto, `columns` na ordem dada)."""
    if fmt == "csv":
        return Response(rows_to_csv(rows, columns), media_type="text/csv")
    return Response(json.dumps(list(rows)), media_type="application/json")


def wide_hist_rows(rows: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    """Expande cada `in_hist_*` (lista de 256) nas colunas `in_hist_*_000..255`."""
    wide: list[dict[str, object]] = []
    for row in rows:
        flat = {key: value for key, value in row.items() if key not in INPUT_HIST_COLUMNS}
        for column in INPUT_HIST_COLUMNS:
            values = row.get(column)
            for i in range(256):
                flat[f"{column}_{i:03d}"] = values[i] if isinstance(values, list) else None
        wide.append(flat)
    return wide


def bad_query_response(exc: Exception) -> JSONResponse:
    return JSONResponse({"detail": str(exc)}, status_code=422)


def variables_endpoint(reader_factory: Callable[[], LogReader]) -> Callable[[Request], Response]:
    """Handler de `GET /data/variables` sobre o leitor dado."""

    def handler(request: Request) -> Response:
        try:
            fmt, limit, include_hist = parse_format(request), parse_limit(request), parse_bool(request.query_params.get("include_hist"), "include_hist")
            since, until = parse_window(request)
        except BadQuery as exc:
            return bad_query_response(exc)
        reader = reader_factory()
        reader.refresh()
        rows = reader.rows(since=since, until=until, limit=limit, include_hist=include_hist)
        if fmt == "csv":
            columns = list(ROW_SCALAR_COLUMNS)
            if include_hist:
                rows = wide_hist_rows(rows)
                columns += hist_csv_columns()
            return table_response(rows, columns, fmt)
        return table_response(rows, [], fmt)

    return handler


def build_data_app(
    reader_factory: Callable[[], LogReader] = get_log_reader, extra_routes: Sequence[Route] = ()
) -> Starlette:
    """App ASGI para montar em `/data`: `/variables` mais as rotas extras do servico."""
    routes = [Route("/variables", variables_endpoint(reader_factory), methods=["GET"]), *extra_routes]
    return Starlette(routes=routes)
