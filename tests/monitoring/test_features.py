from __future__ import annotations

import io

import numpy as np
import pytest
from PIL import Image

from luminus_mlops.monitoring import features as f


def _image(array: np.ndarray) -> Image.Image:
    return Image.fromarray(array.astype(np.uint8), "RGB")


def test_histograms_sum_to_pixel_count_and_means_match_numpy() -> None:
    rng = np.random.default_rng(0)
    img = _image(rng.integers(0, 256, size=(40, 60, 3)))
    feats = f.extract_features(img, file_bytes=123)
    for hist in (feats.hist_l, feats.hist_r, feats.hist_g, feats.hist_b):
        assert len(hist) == 256 and sum(hist) == 40 * 60
    arr = np.asarray(img)
    assert feats.r_mean == pytest.approx(arr[..., 0].mean())
    assert feats.g_mean == pytest.approx(arr[..., 1].mean())
    assert feats.b_mean == pytest.approx(arr[..., 2].mean())
    assert feats.luminance_mean == pytest.approx(np.asarray(img.convert("L")).mean())
    assert feats.file_bytes == 123 and (feats.width, feats.height) == (60, 40)


def test_flat_image_has_no_contrast_and_no_sharpness() -> None:
    flat = f.extract_features(_image(np.full((10, 10, 3), 255)))
    assert flat.sharpness == 0.0 and flat.luminance_std == 0.0


def test_checkerboard_is_sharper_than_a_gradient() -> None:
    checker = (np.indices((32, 32)).sum(axis=0) % 2) * 255
    sharp = f.extract_features(_image(np.stack([checker] * 3, axis=-1)))
    ramp = np.tile(np.linspace(0, 255, 32), (32, 1))
    smooth = f.extract_features(_image(np.stack([ramp] * 3, axis=-1)))
    assert sharp.sharpness > smooth.sharpness and sharp.luminance_std > smooth.luminance_std


def test_uploaded_size_comes_from_the_encoded_file() -> None:
    buffer = io.BytesIO()
    _image(np.zeros((12, 12, 3))).save(buffer, format="PNG")
    opened = Image.open(io.BytesIO(buffer.getvalue()))
    assert f.uploaded_size(opened) == len(buffer.getvalue())


def test_as_dict_follows_column_order_and_uses_lists_for_histograms() -> None:
    d = f.extract_features(_image(np.zeros((3, 3, 3)))).as_dict()
    assert tuple(d) == f.FEATURE_COLUMNS
    assert isinstance(d["hist_l"], list)
