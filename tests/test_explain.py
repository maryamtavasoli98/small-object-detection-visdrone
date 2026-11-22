import numpy as np
import pytest

from smallobj.explain import apply_mask, generate_masks, pixel_order, saliency_stats, similarity
from smallobj.infer import Detections


def test_generate_masks_shape_range_and_density():
    masks = generate_masks(50, (64, 96), grid=8, p=0.5, rng=np.random.default_rng(0))
    assert masks.shape == (50, 64, 96) and masks.dtype == np.float32
    assert masks.min() >= 0.0 and masks.max() <= 1.0
    assert abs(masks.mean() - 0.5) < 0.05  # Bernoulli(0.5) after smoothing keeps the mean


def test_apply_mask_blends_between_image_and_fill():
    img = np.full((4, 4, 3), 200, dtype=np.uint8)
    mask = np.zeros((4, 4), dtype=np.float32)
    mask[:2] = 1.0
    out = apply_mask(img, mask, 0)
    assert out[:2].min() == 200 and out[2:].max() == 0
    fill = np.full_like(img, 50)
    out2 = apply_mask(img, mask * 0.5, fill)
    assert out2[0, 0, 0] == 125


def test_similarity_uses_same_class_iou_times_conf():
    target = np.array([0, 0, 10, 10])
    det = Detections(np.array([[0, 0, 10, 10], [0, 0, 10, 10]], dtype=float), np.array([0.8, 0.9]), np.array([1, 0]))
    assert similarity(target, 0, det) == pytest.approx(0.9)
    assert similarity(target, 1, det) == pytest.approx(0.8)
    assert similarity(target, 2, det) == 0.0
    assert similarity(target, 0, Detections.empty()) == 0.0


def test_pixel_order_saliency_first_random_is_permutation():
    sal = np.array([[0.1, 0.9], [0.5, 0.0]])
    assert pixel_order(sal, "saliency").tolist() == [1, 2, 0, 3]
    rnd = pixel_order(sal, "random", np.random.default_rng(0))
    assert sorted(rnd.tolist()) == [0, 1, 2, 3]


def test_saliency_stats_concentrated_map():
    sal = np.zeros((100, 100), dtype=np.float32)
    sal[40:50, 40:50] = 1.0
    st = saliency_stats(sal, np.array([40, 40, 50, 50]))
    assert st["mass_in_box"] == pytest.approx(1.0)
    assert st["area_frac_90"] == pytest.approx(0.009, abs=0.0015)
    assert st["box_area_frac"] == pytest.approx(0.01)


def test_filter_ignored_regions():
    det = Detections(
        np.array([[0, 0, 10, 10], [100, 100, 110, 110]], dtype=float), np.array([0.5, 0.6]), np.array([0, 0])
    )
    kept = det.filter_ignored([(0, 0, 20, 20)])
    assert len(kept) == 1 and kept.xyxy[0, 0] == 100
