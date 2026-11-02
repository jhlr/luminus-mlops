"""A janela e a referencia usam o mesmo dono da agregacao (`drift.aggregate_histograms`): peso igual por imagem."""

from __future__ import annotations

from pathlib import Path

import pytest

from luminus_mlops.monitoring.analysis import CATEGORICAL_VARIABLES, NUMERIC_VARIABLES, load_reference, window_histograms
from luminus_mlops.monitoring.features import CHANNELS


def _row(channel_bin: int, count: int) -> dict[str, object]:
    hist = [0] * 256
    hist[channel_bin] = count
    return {f"in_hist_{channel}": hist for channel in CHANNELS}


def test_window_histograms_weighs_each_image_equally() -> None:
    hists = window_histograms([_row(10, 1_000_000), _row(200, 100)])
    for channel in CHANNELS:
        assert hists[channel][10] == pytest.approx(1.0)
        assert hists[channel][200] == pytest.approx(1.0)


def test_load_reference_reads_float_histograms(tmp_path: Path) -> None:
    columns = [*NUMERIC_VARIABLES, *CATEGORICAL_VARIABLES]
    (tmp_path / "reference.csv").write_text(",".join(columns) + "\n" + ",".join("1" for _ in columns) + "\n", encoding="utf-8")
    lines = ["bin," + ",".join(CHANNELS), *(f"{i}," + ",".join("0.5" for _ in CHANNELS) for i in range(256))]
    (tmp_path / "reference_hist.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")
    reference = load_reference(tmp_path)
    assert reference.hist["l"][0] == pytest.approx(0.5)
