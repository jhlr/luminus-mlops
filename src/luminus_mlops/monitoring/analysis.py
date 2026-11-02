"""Referencia, resumo da janela e relatorio de drift do ambiente de MONITORAMENTO (SDD secao 7).

Orquestra o que o `monitor` expoe: carrega `artifacts/reference*.csv`, escolhe a janela deslizante
das ultimas N imagens do log (via `log_reader`) e chama a matematica de `drift.py` (KS por canal,
PSI por variavel). Nao define limiares nem formulas: so passa dados para o dono (`drift.py`).
Usa numpy/scipy, entao nunca e importado pelo servico de inferencia.
"""

from __future__ import annotations

import csv
import math
import threading
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .drift import Level, PSIResult, aggregate_histograms, ks_channel, psi_categorical, psi_numeric, worst_level
from .features import CATEGORICAL_FEATURES, CHANNELS, SCALAR_FEATURES
from .log_reader import LogReader
from .outputs import CATEGORICAL_OUTPUTS, NUMERIC_OUTPUTS

DEFAULT_WINDOW_IMAGES = 100
REFRESH_SECONDS = 15.0
REFERENCE_FILE = "reference.csv"
REFERENCE_HIST_FILE = "reference_hist.csv"

# variavel (sem prefixo, como no reference.csv) -> coluna do log
NUMERIC_VARIABLES: dict[str, str] = {
    **{name: f"in_{name}" for name in SCALAR_FEATURES},
    **{name: f"out_{name}" for name in NUMERIC_OUTPUTS},
}
CATEGORICAL_VARIABLES: dict[str, str] = {
    **{name: f"in_{name}" for name in CATEGORICAL_FEATURES},
    **{name: f"out_{name}" for name in CATEGORICAL_OUTPUTS},
}
CHANNEL_LABELS: dict[str, str] = {"L": "l", "R": "r", "G": "g", "B": "b"}


class ReferenceMissing(FileNotFoundError):
    """`artifacts/reference*.csv` ausentes: o monitor nao tem contra o que comparar."""


@dataclass(frozen=True)
class Reference:
    n_images: int
    numeric: dict[str, list[float | None]]
    categorical: dict[str, list[str]]
    hist: dict[str, list[float]]  # canal minusculo -> 256 bins, soma dos histogramas normalizados (total = n imagens)
    table: list[dict[str, str]]  # linhas cruas do reference.csv (para /data/reference)
    hist_table: list[dict[str, str]]  # linhas cruas do reference_hist.csv


def _float_or_none(text: str) -> float | None:
    return float(text) if text != "" else None


def load_reference(directory: Path) -> Reference:
    ref_path, hist_path = directory / REFERENCE_FILE, directory / REFERENCE_HIST_FILE
    if not ref_path.exists() or not hist_path.exists():
        raise ReferenceMissing(
            f"{ref_path} ou {hist_path} ausentes. Rode `just build-reference` (ambiente de inferencia) e suba o monitor de novo."
        )
    with ref_path.open(newline="", encoding="utf-8") as handle:
        table = list(csv.DictReader(handle))
    with hist_path.open(newline="", encoding="utf-8") as handle:
        hist_table = list(csv.DictReader(handle))
    numeric = {name: [_float_or_none(row[name]) for row in table] for name in NUMERIC_VARIABLES}
    categorical = {name: [row[name] for row in table] for name in CATEGORICAL_VARIABLES}
    hist = {channel: [float(row[channel]) for row in hist_table] for channel in CHANNELS}
    return Reference(len(table), numeric, categorical, hist, table, hist_table)


# -- janela -----------------------------------------------------------------------------------
def window_rows(
    reader: LogReader, since: str | None, until: str | None, window_images: int
) -> list[dict[str, object]]:
    """Linhas com variaveis de entrada: as `window_images` mais recentes, ou as de `since`..`until`."""
    bounded = since is not None or until is not None
    return reader.rows(
        since=since, until=until, limit=None if bounded else window_images, include_hist=True, require_input=True
    )


def _numeric_values(rows: Sequence[Mapping[str, object]], column: str) -> list[float | None]:
    values: list[float | None] = []
    for row in rows:
        value = row.get(column)
        values.append(float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else None)
    return values


def _category_values(rows: Sequence[Mapping[str, object]], column: str) -> list[str]:
    return [str(row[column]) for row in rows if row.get(column) is not None]


def window_histograms(rows: Sequence[Mapping[str, object]]) -> dict[str, list[float]]:
    """Histograma 0-255 de cada canal da janela, com peso igual por imagem (`aggregate_histograms`)."""
    return {
        channel: aggregate_histograms([values for row in rows if isinstance(values := row.get(f"in_hist_{channel}"), list)])
        for channel in CHANNELS
    }


# -- drift ------------------------------------------------------------------------------------
@dataclass(frozen=True)
class ChannelDrift:
    channel: str
    d: float | None
    p: float | None
    level: Level
    n_ref: int
    n_win: int


@dataclass(frozen=True)
class DriftReport:
    n_window: int
    n_reference: int
    channels: dict[str, ChannelDrift]
    worst: Level
    psi: dict[str, PSIResult]


def compute_drift(reference: Reference, rows: Sequence[Mapping[str, object]]) -> DriftReport:
    n_win = len(rows)
    hists = window_histograms(rows)
    channels: dict[str, ChannelDrift] = {}
    for label, channel in CHANNEL_LABELS.items():
        if n_win == 0:
            channels[label] = ChannelDrift(label, None, None, Level.INCONCLUSIVE, reference.n_images, 0)
            continue
        result = ks_channel(reference.hist[channel], hists[channel], reference.n_images, n_win)
        channels[label] = ChannelDrift(label, result.d, result.p, result.level, result.n_ref, result.n_win)
    psi: dict[str, PSIResult] = {}
    for name, column in NUMERIC_VARIABLES.items():
        psi[name] = psi_numeric([v for v in reference.numeric[name] if v is not None], [v for v in _numeric_values(rows, column) if v is not None])
    for name, column in CATEGORICAL_VARIABLES.items():
        psi[name] = psi_categorical(reference.categorical[name], _category_values(rows, column))
    return DriftReport(n_win, reference.n_images, channels, worst_level([c.level for c in channels.values()]), psi)


def _finite(value: float | None) -> float | None:
    return None if value is None or math.isnan(value) else value


def drift_to_dict(report: DriftReport, channel: str = "all") -> dict[str, object]:
    wanted = list(CHANNEL_LABELS) if channel == "all" else [channel]
    return {
        "window_images": report.n_window,
        "reference_images": report.n_reference,
        "worst_level": report.worst.name.lower(),
        "channels": {
            label: {
                "d": _finite(report.channels[label].d),
                "p_value": _finite(report.channels[label].p),
                "level": int(report.channels[label].level),
                "level_name": report.channels[label].level.name.lower(),
            }
            for label in wanted
        },
        "psi": {
            name: {"value": _finite(result.value), "bins": result.n_bins, "band": result.band, "reason": result.reason}
            for name, result in report.psi.items()
        },
    }


# -- resumo (/data/summary) -------------------------------------------------------------------
def five_numbers(values: Sequence[float | None]) -> dict[str, float | int] | None:
    """min, Q1, mediana, Q3, max e media para o box plot (`None` se nao ha valores)."""
    clean = np.asarray([v for v in values if v is not None and not math.isnan(v)], dtype=np.float64)
    if clean.size == 0:
        return None
    q1, median, q3 = (float(q) for q in np.quantile(clean, [0.25, 0.5, 0.75]))
    return {
        "n": int(clean.size),
        "min": float(clean.min()),
        "q1": q1,
        "median": median,
        "q3": q3,
        "max": float(clean.max()),
        "mean": float(clean.mean()),
    }


def summarize(reference: Reference, rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    window_hist = window_histograms(rows)
    return {
        "window_images": len(rows),
        "reference_images": reference.n_images,
        "histograms": {
            label: {"window": window_hist[channel], "reference": reference.hist[channel]}
            for label, channel in CHANNEL_LABELS.items()
        },
        "numeric": {
            name: {"window": five_numbers(_numeric_values(rows, column)), "reference": five_numbers(reference.numeric[name])}
            for name, column in NUMERIC_VARIABLES.items()
        },
        "categorical": {
            name: {
                "window": _count(_category_values(rows, column)),
                "reference": _count(reference.categorical[name]),
            }
            for name, column in CATEGORICAL_VARIABLES.items()
        },
    }


def _count(values: Sequence[str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        counts[value] = counts.get(value, 0) + 1
    return counts


# -- monitor (estado do processo) ---------------------------------------------------------------
class DriftMonitor:
    """Le o log de forma incremental e recalcula o drift da janela no maximo a cada `REFRESH_SECONDS`."""

    def __init__(self, reader: LogReader, reference: Reference, window_images: int, refresh_seconds: float = REFRESH_SECONDS) -> None:
        self.reader = reader
        self.reference = reference
        self.window_images = window_images
        self.refresh_seconds = refresh_seconds
        self._lock = threading.Lock()
        self._computed_at: float | None = None
        self._report: DriftReport | None = None

    def current(self) -> DriftReport:
        """Relatorio da janela deslizante; reaproveita o ultimo calculo se tem menos de 15 s."""
        with self._lock:
            now = time.monotonic()
            if self._report is None or self._computed_at is None or now - self._computed_at >= self.refresh_seconds:
                self.reader.refresh()
                rows = window_rows(self.reader, None, None, self.window_images)
                self._report = compute_drift(self.reference, rows)
                self._computed_at = now
            return self._report

    def report(self, since: str | None, until: str | None) -> DriftReport:
        """Com `since`/`until`, calcula na hora para o intervalo; sem eles, a janela deslizante."""
        if since is None and until is None:
            return self.current()
        self.reader.refresh()
        return compute_drift(self.reference, window_rows(self.reader, since, until, self.window_images))

    def summary(self, since: str | None, until: str | None) -> dict[str, object]:
        self.reader.refresh()
        return summarize(self.reference, window_rows(self.reader, since, until, self.window_images))
