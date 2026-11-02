from __future__ import annotations

import numpy as np
import pytest

from luminus_mlops.monitoring import drift


def test_ks_d_identical_zero_and_disjoint_one() -> None:
    hist = [3] * 256
    assert drift.ks_d(hist, hist) == 0.0
    a = [0] * 256
    b = [0] * 256
    a[0], b[255] = 10, 10
    assert drift.ks_d(a, b) == pytest.approx(1.0)


def test_alert_levels_and_inconclusive() -> None:
    assert drift.alert_level(0.04, 20, 20) is drift.Level.GREEN
    assert drift.alert_level(0.05, 20, 20) is drift.Level.YELLOW
    assert drift.alert_level(0.10, 20, 20) is drift.Level.RED
    assert drift.alert_level(0.5, 7, 20) is drift.Level.INCONCLUSIVE
    assert drift.alert_level(0.5, 20, 7) is drift.Level.INCONCLUSIVE
    assert drift.worst_level([drift.Level.GREEN, drift.Level.RED, drift.Level.INCONCLUSIVE]) is drift.Level.RED
    assert drift.worst_level([drift.Level.INCONCLUSIVE]) is drift.Level.INCONCLUSIVE


def test_pvalue_uses_images_not_pixels_and_decreases_with_d() -> None:
    assert drift.ks_pvalue(0.0, 10, 10) == pytest.approx(1.0)
    assert drift.ks_pvalue(0.6, 30, 30) < drift.ks_pvalue(0.2, 30, 30)
    assert 0.0 <= drift.ks_pvalue(0.1, 132, 33) <= 1.0


def test_psi_numeric_same_distribution_small_and_shift_large() -> None:
    rng = np.random.default_rng(1)
    ref = rng.normal(size=2000)
    same = drift.psi_numeric(ref.tolist(), rng.normal(size=2000).tolist())
    shifted = drift.psi_numeric(ref.tolist(), rng.normal(loc=2.0, size=2000).tolist())
    assert same.value is not None and same.value < drift.PSI_MODERATE and same.band == "stable"
    assert shifted.value is not None and shifted.value > drift.PSI_SIGNIFICANT and shifted.band == "significant"
    assert same.n_bins == 10


def test_psi_discrete_and_degenerate_numeric() -> None:
    discrete = drift.psi_numeric([0, 1, 2, 3] * 25, [0, 1, 2, 3] * 5)
    assert discrete.value == pytest.approx(0.0, abs=1e-9) and discrete.n_bins == 4
    constant = drift.psi_numeric([5.0] * 50, [1.0, 2.0])
    assert constant.value is None and constant.reason


def test_psi_categorical_and_histogram() -> None:
    assert drift.psi_categorical(["a", "b"] * 20, ["a", "b"] * 5).value == pytest.approx(0.0, abs=1e-9)
    assert (drift.psi_categorical(["a"] * 40, ["b"] * 10).value or 0) > drift.PSI_SIGNIFICANT
    assert drift.psi_categorical(["a"] * 4, ["a"] * 3).value is None
    ref, win = [1] * 256, [0] * 256
    win[0] = 100
    assert (drift.psi_histogram(ref, win).value or 0) > drift.PSI_SIGNIFICANT
    assert drift.regroup_histogram(ref).tolist() == [16.0] * 16


def test_benjamini_hochberg_is_monotone_and_not_smaller() -> None:
    p = [0.001, 0.2, 0.03, 0.04]
    adj = drift.benjamini_hochberg(p)
    assert all(a >= b for a, b in zip(adj, p)) and max(adj) <= 1.0


def _spike(bin_index: int, count: int) -> list[int]:
    hist = [0] * 256
    hist[bin_index] = count
    return hist


def test_aggregate_histograms_gives_every_image_the_same_weight() -> None:
    # uma foto enorme (1.000.000 de pixels) e uma pequena (100 pixels) em bins diferentes
    total = drift.aggregate_histograms([_spike(10, 1_000_000), _spike(200, 100)])
    assert total[10] == pytest.approx(1.0)
    assert total[200] == pytest.approx(1.0)
    assert sum(total) == pytest.approx(2.0)  # total = numero de imagens


def test_aggregate_histograms_huge_image_does_not_dominate_the_ks() -> None:
    reference = drift.aggregate_histograms([_spike(10, 100)] * 5 + [_spike(200, 100)] * 5)
    # janela com a mesma mistura 50/50 de imagens, mas a de bin 10 e 10.000x maior
    window = drift.aggregate_histograms([_spike(10, 1_000_000)] * 5 + [_spike(200, 100)] * 5)
    assert drift.ks_d(reference, window) == pytest.approx(0.0)
    # com a soma crua de pixels a foto grande dominaria: D proximo de 0.5
    raw = np.sum([_spike(10, 1_000_000)] * 5 + [_spike(200, 100)] * 5, axis=0)
    assert drift.ks_d(reference, raw.tolist()) > 0.4


def test_aggregate_histograms_ignores_empty_histograms_and_empty_input() -> None:
    assert sum(drift.aggregate_histograms([_spike(3, 7), [0] * 256])) == pytest.approx(1.0)
    assert drift.aggregate_histograms([]) == [0.0] * 256
