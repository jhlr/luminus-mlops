"""Reference x holdout split and traffic image lists (single owner, SDD Decision 7).

Dataset: Imagenette 320px (fast.ai, 10 easy ImageNet classes). The reference is a deterministic
sample of its `train` images: sorted paths, shuffled with seed 0, first `N_REFERENCE`. The other
`train` images are the `holdout` (the no-drift control). Shared by `scripts/build_reference.py` and
`scripts/load_gen.py`.
"""

from __future__ import annotations

import random
from pathlib import Path

from ..paths import DATA_DIR

IMAGENETTE_DIR = DATA_DIR / "imagenette2-320"
IMAGEWOOF_DIR = DATA_DIR / "imagewoof2-320"
N_REFERENCE = 500
DEFAULT_SEED = 0
TRAFFIC_SETS: tuple[str, ...] = ("holdout", "val", "woof")
IMAGE_SUFFIXES = frozenset({".jpeg", ".jpg", ".png"})


def list_images(directory: Path) -> list[Path]:
    """Sorted images (JPEG/PNG) of a folder, recursive (class sub-folders, `samples/`...)."""
    return sorted(p for p in directory.rglob("*") if p.suffix.lower() in IMAGE_SUFFIXES)


def split_reference_images(
    dataset_dir: Path = IMAGENETTE_DIR, seed: int = DEFAULT_SEED, n_reference: int = N_REFERENCE
) -> tuple[list[Path], list[Path]]:
    """Return (reference, holdout) from the `train` images of `dataset_dir`."""
    names = list_images(dataset_dir / "train")
    if len(names) <= n_reference:
        raise ValueError(f"{dataset_dir / 'train'} has {len(names)} images, not more than the reference ({n_reference})")
    shuffled = list(names)
    random.Random(seed).shuffle(shuffled)
    return shuffled[:n_reference], shuffled[n_reference:]


def traffic_images(traffic: str, seed: int = DEFAULT_SEED, data_dir: Path | None = None) -> list[Path]:
    """Images of a traffic set: `holdout` (train, not in the reference), `val` (other split) or `woof` (other dataset)."""
    base = data_dir if data_dir is not None else DATA_DIR
    if traffic == "holdout":
        return split_reference_images(base / IMAGENETTE_DIR.name, seed)[1]
    if traffic == "val":
        return list_images(base / IMAGENETTE_DIR.name / "val")
    if traffic == "woof":
        return list_images(base / IMAGEWOOF_DIR.name / "val")
    raise ValueError(f"unknown traffic set: {traffic!r} (use one of {TRAFFIC_SETS})")
