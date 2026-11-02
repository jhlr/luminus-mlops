from __future__ import annotations

from pathlib import Path

import pytest

from luminus_mlops.monitoring import split


def _make_dataset(root: Path, n: int = 40) -> Path:
    for i in range(n):
        folder = root / "train" / f"n{i % 4:03d}"
        folder.mkdir(parents=True, exist_ok=True)
        (folder / f"img_{i:04d}.JPEG").write_bytes(b"x")
    (root / "train" / "notes.txt").write_text("ignored")
    return root


def test_split_is_deterministic_disjoint_and_ignores_other_files(tmp_path: Path) -> None:
    dataset = _make_dataset(tmp_path)
    ref1, hold1 = split.split_reference_images(dataset, seed=0, n_reference=30)
    ref2, hold2 = split.split_reference_images(dataset, seed=0, n_reference=30)
    assert ref1 == ref2 and hold1 == hold2
    assert len(ref1) == 30 and len(hold1) == 10 and not set(ref1) & set(hold1)
    assert split.split_reference_images(dataset, seed=1, n_reference=30)[0] != ref1


def test_too_few_images_is_an_error(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        split.split_reference_images(_make_dataset(tmp_path, 5))


def test_unknown_traffic_is_an_error() -> None:
    with pytest.raises(ValueError):
        split.traffic_images("nope")


def test_real_dataset_counts_when_present() -> None:
    if not split.IMAGENETTE_DIR.exists():
        pytest.skip("data/ missing (just data)")
    reference, holdout = split.split_reference_images()
    assert len(reference) == split.N_REFERENCE and not set(reference) & set(holdout)
    assert split.traffic_images("holdout") == holdout
    assert split.traffic_images("val") and not set(split.traffic_images("val")) & set(reference)
