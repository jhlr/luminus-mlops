"""Guardas estaticas: um dono por responsabilidade (features, limiares de drift, split)."""

from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MON = ROOT / "src" / "luminus_mlops" / "monitoring"
CONSUMERS = [
    ROOT / "src" / "luminus_mlops" / "service.py",
    ROOT / "src" / "luminus_mlops" / "monitor_service.py",
    *[p for p in MON.glob("*.py")],
    *[ROOT / "scripts" / n for n in ("build_reference.py", "load_gen.py", "analyze_log.py", "make_metrics_xlsx.py")],
]


def _files_except(owner: str) -> list[Path]:
    return [p for p in CONSUMERS if p.exists() and p.name != owner]


def _trees(paths: list[Path]) -> list[tuple[Path, ast.AST]]:
    return [(p, ast.parse(p.read_text(encoding="utf-8"))) for p in paths]


def test_owner_files_exist() -> None:
    for name in ("features.py", "drift.py", "split.py", "outputs.py", "log_reader.py", "logging_setup.py", "metrics.py"):
        assert (MON / name).exists(), name


def test_features_are_computed_only_in_features_py() -> None:
    offenders: list[str] = []
    for path, tree in _trees(_files_except("features.py")):
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                func = node.func
                name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", "")
                if name in {"histogram", "Laplacian", "ImageFeatures"}:
                    offenders.append(f"{path.name}:{node.lineno} chama {name}")
    assert not offenders, offenders


def test_drift_thresholds_live_only_in_drift_py() -> None:
    owned_names = {"D_YELLOW", "D_RED", "MIN_IMAGES", "PSI_MODERATE", "PSI_SIGNIFICANT"}
    magic = {0.05, 0.1, 0.25}
    offenders: list[str] = []
    for path, tree in _trees(_files_except("drift.py")):
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id in owned_names:
                        offenders.append(f"{path.name}:{node.lineno} define {target.id}")
            if isinstance(node, ast.Compare):
                for operand in [node.left, *node.comparators]:
                    if isinstance(operand, ast.Constant) and operand.value in magic and isinstance(operand.value, float):
                        offenders.append(f"{path.name}:{node.lineno} compara com {operand.value}")
    assert not offenders, offenders


def test_reference_split_lives_only_in_split_py() -> None:
    offenders: list[str] = []
    for path, tree in _trees(_files_except("split.py")):
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "shuffle":
                offenders.append(f"{path.name}:{node.lineno} embaralha")
            if isinstance(node, ast.Constant) and isinstance(node.value, str) and "imagenette2" in node.value:
                offenders.append(f"{path.name}:{node.lineno} nome do dataset de referencia")
    assert not offenders, offenders
