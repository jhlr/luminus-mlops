"""Leitura do log de servico: arquivo atual + rotacionados -> uma linha por `request_id`.

Dono unico do parse (usado pelo servico de inferencia e pelo `monitor`); so biblioteca padrao,
porque a inferencia nao tem pandas. A leitura e INCREMENTAL: guarda o offset em bytes por inode
(rotacao = o inode antigo continua sendo lido de onde parou, via `inference.log.N`), le so os
bytes novos e mantem a janela deslizante num deque limitado. Nunca reler ate 300 MB a cada
atualizacao.
"""

from __future__ import annotations

import json
import os
import re
import threading
from array import array
from collections import OrderedDict, deque
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from .features import CATEGORICAL_FEATURES, HIST_FEATURES, SCALAR_FEATURES
from .logging_setup import LOG_FILE_NAME, utc_iso
from .outputs import OUTPUT_COLUMNS

DEFAULT_MAX_ROWS = 10_000
DEFAULT_MAX_PENDING = 1_000
FILE_PATTERN = re.compile(rf"^{re.escape(LOG_FILE_NAME)}(?:\.(\d+))?$")

INPUT_SCALAR_COLUMNS: tuple[str, ...] = tuple(f"in_{n}" for n in SCALAR_FEATURES + CATEGORICAL_FEATURES)
INPUT_HIST_COLUMNS: tuple[str, ...] = tuple(f"in_{n}" for n in HIST_FEATURES)
OUTPUT_ROW_COLUMNS: tuple[str, ...] = tuple(f"out_{n}" for n in OUTPUT_COLUMNS)
ROW_SCALAR_COLUMNS: tuple[str, ...] = (
    "ts",
    "request_id",
    "endpoint",
    "method",
    "status_code",
    "latency_ms",
    "error",
    *INPUT_SCALAR_COLUMNS,
    *OUTPUT_ROW_COLUMNS,
)


@dataclass
class ReaderStats:
    lines_parsed: int = 0
    bad_lines: int = 0
    bytes_read: int = 0


def normalize_ts(value: str | datetime | None) -> str | None:
    """ISO 8601 (qualquer offset) -> string UTC de largura fixa, comparavel como texto."""
    if value is None:
        return None
    moment = datetime.fromisoformat(value) if isinstance(value, str) else value
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=timezone.utc)
    return utc_iso(moment.timestamp())


class LogReader:
    def __init__(
        self,
        directory: Path,
        max_rows: int | None = None,
        max_pending: int = DEFAULT_MAX_PENDING,
    ) -> None:
        self.directory = directory
        rows_cap = max_rows if max_rows is not None else int(os.environ.get("LOG_READER_MAX_ROWS", DEFAULT_MAX_ROWS))
        self._rows: deque[dict[str, object]] = deque(maxlen=rows_cap)
        self._pending: OrderedDict[str, dict[str, object]] = OrderedDict()
        self._max_pending = max_pending
        self._offsets: dict[int, int] = {}  # inode -> bytes ja consumidos
        self._lock = threading.RLock()  # handlers sincronos do BentoML rodam em threads
        self.stats = ReaderStats()

    # -- arquivos --------------------------------------------------------------
    def _files(self) -> list[Path]:
        """Rotacionados do mais antigo (`.5`) ao mais novo (`.1`) e por ultimo o ativo."""
        found: list[tuple[int, Path]] = []
        if self.directory.is_dir():
            for path in self.directory.iterdir():
                match = FILE_PATTERN.match(path.name)
                if match:
                    found.append((int(match.group(1)) if match.group(1) else 0, path))
        found.sort(key=lambda item: -item[0] if item[0] else 1)
        return [path for _, path in found]

    @property
    def offsets(self) -> dict[int, int]:
        return dict(self._offsets)

    # -- parse -----------------------------------------------------------------
    def refresh(self) -> int:
        """Le so os bytes novos de cada arquivo; devolve quantas linhas novas foram parseadas."""
        with self._lock:
            return self._refresh()

    def _refresh(self) -> int:
        parsed_before = self.stats.lines_parsed
        live_inodes: set[int] = set()
        for path in self._files():
            try:
                stat = path.stat()
            except FileNotFoundError:  # rotacao descartou o arquivo entre a listagem e o stat
                continue
            inode = stat.st_ino
            live_inodes.add(inode)
            offset = self._offsets.get(inode, 0)
            if stat.st_size < offset:  # truncado: recomeca
                offset = 0
            if stat.st_size == offset:
                self._offsets[inode] = offset
                continue
            try:
                with path.open("rb") as handle:
                    handle.seek(offset)
                    chunk = handle.read(stat.st_size - offset)
            except FileNotFoundError:
                continue
            end = chunk.rfind(b"\n") + 1  # so linhas completas; a ultima parcial fica para depois
            for raw in chunk[:end].splitlines():
                self._ingest(raw)
            self._offsets[inode] = offset + end
            self.stats.bytes_read += end
        for inode in set(self._offsets) - live_inodes:
            del self._offsets[inode]
        return self.stats.lines_parsed - parsed_before

    def _ingest(self, raw: bytes) -> None:
        if not raw.strip():
            return
        try:
            event = json.loads(raw)
        except json.JSONDecodeError:
            self.stats.bad_lines += 1
            return
        self.stats.lines_parsed += 1
        request_id = str(event.get("request_id"))
        if event.get("event") == "request":
            self._pending[request_id] = event
            while len(self._pending) > self._max_pending:
                self._pending.popitem(last=False)
        elif event.get("event") == "response":
            self._rows.append(self._join(self._pending.pop(request_id, None), event))

    @staticmethod
    def _join(request: dict[str, object] | None, response: dict[str, object]) -> dict[str, object]:
        row: dict[str, object] = {
            "ts": (request or response).get("ts"),
            "request_id": response.get("request_id"),
            "endpoint": response.get("endpoint"),
            "method": response.get("method"),
            "status_code": response.get("status_code"),
            "latency_ms": response.get("latency_ms"),
            "error": response.get("error"),
        }
        source = request or {}
        for column in INPUT_SCALAR_COLUMNS:
            row[column] = source.get(column)
        for column in INPUT_HIST_COLUMNS:
            values = source.get(column)
            row[column] = array("I", values) if isinstance(values, list) else None
        for column in OUTPUT_ROW_COLUMNS:
            row[column] = response.get(column)
        return row

    # -- consulta --------------------------------------------------------------
    def rows(
        self,
        since: str | datetime | None = None,
        until: str | datetime | None = None,
        limit: int | None = None,
        include_hist: bool = False,
        require_input: bool = False,
    ) -> list[dict[str, object]]:
        """Linhas da janela, da mais antiga a mais nova, ficando com as `limit` mais recentes.

        Com `include_hist` os 4 histogramas voltam como listas; sem ele as colunas `in_hist_*` saem.
        `require_input` descarta as linhas sem evento `request` (sem variaveis de entrada).
        """
        low, high = normalize_ts(since), normalize_ts(until)
        with self._lock:
            selected = [
                row
                for row in self._rows
                if (low is None or str(row["ts"]) >= low)
                and (high is None or str(row["ts"]) <= high)
                and (not require_input or row[INPUT_HIST_COLUMNS[0]] is not None)
            ]
            return self._copy_rows(selected, limit, include_hist)

    @staticmethod
    def _copy_rows(selected: list[dict[str, object]], limit: int | None, include_hist: bool) -> list[dict[str, object]]:
        if limit is not None:
            selected = selected[-limit:] if limit > 0 else []
        result: list[dict[str, object]] = []
        for row in selected:
            copy = dict(row)
            for column in INPUT_HIST_COLUMNS:
                hist = copy[column]
                if include_hist:
                    copy[column] = list(hist) if isinstance(hist, array) else None
                else:
                    del copy[column]
            result.append(copy)
        return result
