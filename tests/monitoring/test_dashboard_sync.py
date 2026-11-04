"""O dashboard versionado tem que ser exatamente o que `scripts/build_dashboards.py` gera hoje."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import ModuleType

ROOT = Path(__file__).resolve().parents[2]


def _load_builder() -> ModuleType:
    spec = importlib.util.spec_from_file_location("build_dashboards", ROOT / "scripts" / "build_dashboards.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_committed_dashboard_matches_generator() -> None:
    builder = _load_builder()
    board = builder.build_dashboard()
    committed = json.loads((builder.OUT_DIR / f"{board.uid}.json").read_text(encoding="utf-8"))
    assert committed == board.dump(), "rode `uv run python scripts/build_dashboards.py` e versione o JSON"
