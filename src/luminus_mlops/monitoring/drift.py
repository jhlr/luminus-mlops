"""KS, PSI e limiares de alerta (dono unico da matematica de drift, SDD secoes 3, 7 e 9).

KS por canal (metodo do curso): D = max |F_ref - F_janela| sobre as CDFs dos histogramas
0-255 agregados com PESO IGUAL por imagem (`aggregate_histograms`: cada histograma e normalizado
para somar 1 antes da soma, senao as fotos grandes dominam). O p-valor usa `scipy.stats.kstwobign` com `n_eff` calculado em IMAGENS (nao
pixels, que dentro de uma foto nao sao independentes). D e a grandeza monitorada.

PSI: 10 bins por quantis da referencia para numericas, 16 bins (regrupados de 256) para os
histogramas de pixels e as proprias categorias para variaveis categoricas; epsilon 1e-6.
As faixas de PSI sao convencao da industria, nao regra provada.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Hashable, Iterable, Sequence
from dataclasses import dataclass
from enum import IntEnum

import numpy as np
from scipy import stats

from .features import HIST_BINS

D_YELLOW = 0.05
D_RED = 0.10
MIN_IMAGES = 8
PSI_MODERATE = 0.10
PSI_SIGNIFICANT = 0.25
PSI_EPSILON = 1e-6
PSI_QUANTILE_BINS = 10
PSI_HIST_BINS = 16
DISCRETE_MAX_DISTINCT = 10


class Level(IntEnum):
    """Valores usados direto no gauge `luminus_drift_level`."""

    INCONCLUSIVE = -1
    GREEN = 0
    YELLOW = 1
    RED = 2


@dataclass(frozen=True)
class KSResult:
    d: float
    p: float
    level: Level
    n_ref: int
    n_win: int


@dataclass(frozen=True)
class PSIResult:
    value: float | None
    n_bins: int
    reason: str | None = None

    @property
    def band(self) -> str | None:
        return None if self.value is None else psi_band(self.value)


def aggregate_histograms(hists: Iterable[Sequence[float]]) -> list[float]:
    """Dono unico da agregacao: soma dos histogramas de varias imagens com peso igual por imagem.

    Cada histograma (256 bins) e normalizado para somar 1 e depois somado, entao o total e o
    numero de imagens e uma foto de milhoes de pixels pesa o mesmo que uma pequena. Histogramas
    vazios (soma 0) sao ignorados; sem imagens devolve 256 zeros. Usado pela referencia
    (`reference_hist.csv`), pela janela, pelo `/data/summary`, pelos gauges e pelo `analyze_log`.
    """
    total = np.zeros(HIST_BINS, dtype=np.float64)
    for hist in hists:
        arr = np.asarray(hist, dtype=np.float64)
        if arr.shape != total.shape:
            raise ValueError("histograma com tamanho diferente de 256 bins")
        mass = arr.sum()
        if mass > 0:
            total += arr / mass
    return [float(v) for v in total]


def ks_d(ref_hist: Sequence[float], win_hist: Sequence[float]) -> float:
    """D = max |F_ref - F_janela| entre as CDFs de dois histogramas de mesmo tamanho."""
    ref = np.asarray(ref_hist, dtype=np.float64)
    win = np.asarray(win_hist, dtype=np.float64)
    if ref.shape != win.shape:
        raise ValueError("histogramas com tamanhos diferentes")
    if ref.sum() == 0 or win.sum() == 0:
        raise ValueError("histograma vazio")
    return float(np.max(np.abs(np.cumsum(ref) / ref.sum() - np.cumsum(win) / win.sum())))


def ks_pvalue(d: float, n_ref: int, n_win: int) -> float:
    """p-valor assintotico: sobrevivencia de kstwobign em sqrt(n_eff) * D (n em imagens)."""
    n_eff = n_ref * n_win / (n_ref + n_win)
    return float(stats.kstwobign.sf(math.sqrt(n_eff) * d))


def alert_level(d: float, n_ref: int, n_win: int) -> Level:
    if n_ref < MIN_IMAGES or n_win < MIN_IMAGES:
        return Level.INCONCLUSIVE
    if d >= D_RED:
        return Level.RED
    if d >= D_YELLOW:
        return Level.YELLOW
    return Level.GREEN


def ks_channel(ref_hist: Sequence[float], win_hist: Sequence[float], n_ref: int, n_win: int) -> KSResult:
    d = ks_d(ref_hist, win_hist)
    return KSResult(d=d, p=ks_pvalue(d, n_ref, n_win), level=alert_level(d, n_ref, n_win), n_ref=n_ref, n_win=n_win)


def worst_level(levels: Sequence[Level]) -> Level:
    """Pior nivel dos canais (INCONCLUSIVE so se todos forem inconclusivos)."""
    known = [level for level in levels if level != Level.INCONCLUSIVE]
    return max(known) if known else Level.INCONCLUSIVE


def scalar_ks(ref: Sequence[float], win: Sequence[float]) -> tuple[float, float]:
    """KS de duas amostras de escalares (n = imagens): (estatistica, p-valor)."""
    result = stats.ks_2samp(np.asarray(ref, dtype=np.float64), np.asarray(win, dtype=np.float64))
    return float(result.statistic), float(result.pvalue)


def psi_band(value: float) -> str:
    if value < PSI_MODERATE:
        return "stable"
    if value <= PSI_SIGNIFICANT:
        return "moderate"
    return "significant"


def _psi_from_counts(ref_counts: np.ndarray, win_counts: np.ndarray) -> float:
    ref_p = np.maximum(ref_counts / ref_counts.sum(), PSI_EPSILON)
    win_p = np.maximum(win_counts / win_counts.sum(), PSI_EPSILON)
    return float(np.sum((win_p - ref_p) * np.log(win_p / ref_p)))


def psi_numeric(ref: Sequence[float], win: Sequence[float]) -> PSIResult:
    """PSI de uma variavel numerica; bins por quantis da referencia.

    Variaveis discretas ou degeneradas: bordas repetidas sao descartadas; com no maximo 10
    valores distintos na referencia usa-se um bin por valor; com menos de 2 bins (referencia
    constante) devolve `value=None` com o motivo.
    """
    ref_arr = np.asarray([v for v in ref if v is not None and not math.isnan(v)], dtype=np.float64)
    win_arr = np.asarray([v for v in win if v is not None and not math.isnan(v)], dtype=np.float64)
    if ref_arr.size == 0 or win_arr.size == 0:
        return PSIResult(None, 0, "sem dados na referencia ou na janela")
    distinct = np.unique(ref_arr)
    if distinct.size < 2:
        return PSIResult(None, distinct.size, "referencia constante (menos de 2 bins)")
    if distinct.size <= DISCRETE_MAX_DISTINCT:
        index = {float(v): i for i, v in enumerate(distinct)}
        ref_counts = np.zeros(distinct.size)
        win_counts = np.zeros(distinct.size)
        for v in ref_arr:
            ref_counts[index[float(v)]] += 1
        for v in win_arr:
            # valor fora do suporte da referencia cai no bin mais proximo
            win_counts[int(np.argmin(np.abs(distinct - v)))] += 1
        return PSIResult(_psi_from_counts(ref_counts, win_counts), int(distinct.size))
    edges = np.unique(np.quantile(ref_arr, np.linspace(0, 1, PSI_QUANTILE_BINS + 1)))
    if edges.size < 3:
        return PSIResult(None, int(edges.size) - 1, "bordas de quantil repetidas (menos de 2 bins)")
    inner = edges[1:-1]
    ref_counts = np.bincount(np.searchsorted(inner, ref_arr, side="right"), minlength=edges.size - 1).astype(np.float64)
    win_counts = np.bincount(np.searchsorted(inner, win_arr, side="right"), minlength=edges.size - 1).astype(np.float64)
    return PSIResult(_psi_from_counts(ref_counts, win_counts), int(edges.size) - 1)


def psi_categorical(ref: Sequence[Hashable], win: Sequence[Hashable]) -> PSIResult:
    """PSI sobre as categorias (uniao das categorias da referencia e da janela)."""
    ref_counter, win_counter = Counter(ref), Counter(win)
    if not ref_counter or not win_counter:
        return PSIResult(None, 0, "sem dados na referencia ou na janela")
    categories = sorted(set(ref_counter) | set(win_counter), key=str)
    if len(categories) < 2:
        return PSIResult(None, len(categories), "uma unica categoria (menos de 2 bins)")
    ref_counts = np.array([ref_counter[c] for c in categories], dtype=np.float64)
    win_counts = np.array([win_counter[c] for c in categories], dtype=np.float64)
    return PSIResult(_psi_from_counts(ref_counts, win_counts), len(categories))


def regroup_histogram(hist: Sequence[float], bins: int = PSI_HIST_BINS) -> np.ndarray:
    """Regrupa um histograma (256 bins) em `bins` faixas somando bins vizinhos."""
    arr = np.asarray(hist, dtype=np.float64)
    if arr.size % bins:
        raise ValueError("tamanho do histograma nao e multiplo do numero de bins")
    return arr.reshape(bins, -1).sum(axis=1)


def psi_histogram(ref_hist: Sequence[float], win_hist: Sequence[float]) -> PSIResult:
    """PSI dos histogramas de pixels agregados, regrupados de 256 para 16 bins."""
    ref16, win16 = regroup_histogram(ref_hist), regroup_histogram(win_hist)
    if ref16.sum() == 0 or win16.sum() == 0:
        return PSIResult(None, 0, "histograma vazio")
    return PSIResult(_psi_from_counts(ref16, win16), PSI_HIST_BINS)


def benjamini_hochberg(p_values: Sequence[float]) -> list[float]:
    """p-valores ajustados por Benjamini-Hochberg (FDR)."""
    result = stats.false_discovery_control(np.asarray(p_values, dtype=np.float64), method="bh")
    return [float(p) for p in result]
