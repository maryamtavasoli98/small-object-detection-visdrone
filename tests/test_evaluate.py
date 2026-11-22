import json

import numpy as np
import pytest

from smallobj.evaluate import SIZE_BINS, evaluate_coco, match_predictions


def _gt(tmp_path):
    images = [{"id": 1, "file_name": "a.jpg", "width": 1000, "height": 1000}]
    anns = []
    sizes = [(10, 10), (20, 20), (50, 50), (150, 150)]  # tiny, small, medium, large
    for i, (w, h) in enumerate(sizes, start=1):
        x, y = 100 * i, 100 * i
        anns.append({"id": i, "image_id": 1, "category_id": 1, "bbox": [x, y, w, h], "area": w * h, "iscrowd": 0})
    gt = {
        "images": images,
        "annotations": anns,
        "categories": [{"id": 1, "name": "pedestrian"}, {"id": 2, "name": "people"}],
    }
    p = tmp_path / "gt.json"
    p.write_text(json.dumps(gt))
    return p, anns


def test_perfect_predictions_give_ap_one_in_every_bin(tmp_path):
    gt_path, anns = _gt(tmp_path)
    preds = [{"image_id": 1, "category_id": 1, "bbox": a["bbox"], "score": 0.9} for a in anns]
    m = evaluate_coco(gt_path, preds)
    assert m["overall"]["AP"] == pytest.approx(1.0)
    for b in ("tiny", "small", "medium", "large"):
        assert m["by_size"][b]["AP50"] == pytest.approx(1.0)
        assert m["by_size"][b]["n_gt"] == 1
    assert m["by_size"]["coco_small"]["n_gt"] == 2
    assert m["per_class"]["pedestrian"]["AP50"] == pytest.approx(1.0)
    assert np.isnan(m["per_class"]["people"]["AP50"])  # no GT of that class


def test_missing_tiny_object_only_hurts_tiny_bin(tmp_path):
    gt_path, anns = _gt(tmp_path)
    preds = [{"image_id": 1, "category_id": 1, "bbox": a["bbox"], "score": 0.9} for a in anns[1:]]
    m = evaluate_coco(gt_path, preds)
    assert m["by_size"]["tiny"]["AP50"] == pytest.approx(0.0)
    assert m["by_size"]["large"]["AP50"] == pytest.approx(1.0)
    assert m["overall"]["AP50"] == pytest.approx(0.75, abs=0.01)


def test_size_bins_are_contiguous():
    labels = ["tiny", "small", "medium", "large"]
    for a, b in zip(labels, labels[1:], strict=False):
        assert SIZE_BINS[a][1] == SIZE_BINS[b][0]


def test_match_predictions_greedy_by_confidence():
    gt = np.array([[0, 0, 10, 10], [50, 50, 60, 60]], dtype=float)
    gt_cls = np.array([0, 1])
    det = np.array([[0, 0, 10, 10], [1, 1, 10, 10], [50, 50, 60, 60]], dtype=float)
    det_cls = np.array([0, 0, 0])
    conf = np.array([0.5, 0.9, 0.9])
    idx = match_predictions(gt, gt_cls, det, det_cls, conf, iou_thr=0.5)
    assert idx.tolist() == [-1, 0, -1]  # higher-conf duplicate wins; wrong class never matches
