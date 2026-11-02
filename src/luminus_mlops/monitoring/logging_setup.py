"""Logger JSON Lines do servico (SDD Decisao 4, secao 5.4).

O handler (`RotatingFileHandler`) e criado de forma preguicosa DENTRO do processo worker, nunca
no import do modulo: `bentoml serve` faz fork e a documentacao do BentoML nao suporta o handler
na definicao do servico. O arquivo e append-only; so a rotacao renomeia (50 MB, 5 backups).
Um unico worker escreve (o handler nao e seguro entre processos).
"""

from __future__ import annotations

import hashlib
import json
import logging
import logging.handlers
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from ..paths import ROOT

LOGGER_NAME = "luminus.serving"
LOG_FILE_NAME = "inference.log"
DEFAULT_LOG_DIR = "./serving-logs"
DEFAULT_MAX_BYTES = 50 * 1024 * 1024
BACKUP_COUNT = 5
IP_HASH_CHARS = 12


def utc_iso(epoch_seconds: float) -> str:
    """ISO 8601 em UTC com microssegundos e offset explicito (`...481516+00:00`)."""
    return datetime.fromtimestamp(epoch_seconds, tz=timezone.utc).isoformat(timespec="microseconds")


def hash_client_ip(ip: str | None) -> str | None:
    """Primeiros 12 hex do SHA-256 do IP: nunca o IP cru."""
    if not ip:
        return None
    return hashlib.sha256(ip.encode("utf-8")).hexdigest()[:IP_HASH_CHARS]


def service_version() -> str:
    """`SERVICE_VERSION` do ambiente; sem ele tenta o git local; senao `unknown`."""
    env = os.environ.get("SERVICE_VERSION")
    if env:
        return env
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True, timeout=5
        )
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    return out.stdout.strip() or "unknown"


def log_dir() -> Path:
    return Path(os.environ.get("SERVING_LOG_DIR", DEFAULT_LOG_DIR))


def max_bytes() -> int:
    return int(os.environ.get("SERVING_LOG_MAX_BYTES", DEFAULT_MAX_BYTES))


class JsonFormatter(logging.Formatter):
    """Uma linha = um objeto JSON; nada de mensagem livre."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, object] = {"ts": utc_iso(record.created), "level": record.levelname}
        payload.update(getattr(record, "fields", {}))
        return json.dumps(payload)


def get_serving_logger() -> logging.Logger:
    """Logger `luminus.serving` (INFO, sem propagacao), com o handler criado na 1a chamada."""
    logger = logging.getLogger(LOGGER_NAME)
    if not logger.handlers:
        directory = log_dir()
        directory.mkdir(parents=True, exist_ok=True)
        handler = logging.handlers.RotatingFileHandler(
            directory / LOG_FILE_NAME, mode="a", maxBytes=max_bytes(), backupCount=BACKUP_COUNT, encoding="utf-8"
        )
        handler.setFormatter(JsonFormatter())
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False
    return logger


def write_event(level: int, fields: dict[str, object]) -> None:
    """Grava um evento. Falhas de escrita ficam com `logging.handleError`, nunca derrubam a requisicao."""
    get_serving_logger().log(level, fields["event"], extra={"fields": fields})
