"""COCO-style evaluation with custom object-size bins (pycocotools).

COCO reports AP for small (< 32^2 px), medium and large (> 96^2 px) objects. Drone imagery
needs a finer look at the low end, so we add a *tiny* bin (< 16^2 px) and report AP, AP50 and
AR per bin and per class. Areas are measured in pixels of the original image.
"""

from __future__ import annotations

import contextlib
import io
import json
from pathlib import Path

import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

INF = 1e10
SIZE_BINS: dict[str, list[float]] = {
    "all": [0.0, INF],
    "tiny": [0.0, 16.0**2],
    "small": [16.0**2, 32.0**2],
    "medium": [32.0**2, 96.0**2],
    "large": [96.0**2, INF],
    "coco_small": [0.0, 32.0**2],
}


def _mean_valid(x: np.ndarray) -> float:
    x = x[x > -1]
    return float(x.mean()) if x.size else float("nan")


def count_gt_by_bin(coco_gt: COCO, bins: dict[str, list[float]]) -> dict[str, int]:
    areas = np.array([a["area"] for a in coco_gt.dataset["annotations"]], dtype=np.float64)
    return {name: int(((areas >= lo) & (areas < hi)).sum()) for name, (lo, hi) in bins.items()}


def evaluate_coco(
    gt_json: Path,
    predictions: Path | list[dict],
    bins: dict[str, list[float]] | None = None,
    max_dets: int = 1000,
) -> dict:
    """Evaluate COCO-format predictions and return a nested dict of metrics.

    Returned keys: ``overall`` (AP, AP50, AP75, AR), ``by_size`` (per bin: AP, AP50, AR, n_gt),
    ``per_class`` (AP, AP50 per class name) and ``per_class_by_size`` (AP50 per class and bin).
    """
    bins = bins or SIZE_BINS
    with contextlib.redirect_stdout(io.StringIO()):
        coco_gt = COCO(str(gt_json))
        preds = json.loads(Path(predictions).read_text()) if isinstance(predictions, (str, Path)) else predictions
        if len(preds) == 0:
            raise ValueError("no predictions to evaluate")
        coco_dt = coco_gt.loadRes(preds)
        ev = COCOeval(coco_gt, coco_dt, "bbox")
        ev.params.areaRng = [list(v) for v in bins.values()]
        ev.params.areaRngLbl = list(bins.keys())
        ev.params.maxDets = [max_dets]
        ev.evaluate()
        ev.accumulate()

    precision = ev.eval["precision"]  # [T, R, K, A, M]
    recall = ev.eval["recall"]  # [T, K, A, M]
    cat_ids = ev.params.catIds
    names = {c["id"]: c["name"] for c in coco_gt.dataset["categories"]}
    t50 = int(np.argmin(np.abs(ev.params.iouThrs - 0.5)))
    t75 = int(np.argmin(np.abs(ev.params.iouThrs - 0.75)))
    n_gt = count_gt_by_bin(coco_gt, bins)

    def ap(a: int, t: int | None = None, k: int | None = None) -> float:
        p = precision[:, :, :, a, 0]
        if t is not None:
            p = p[t : t + 1]
        if k is not None:
            p = p[:, :, k : k + 1]
        return _mean_valid(p)

    def ar(a: int, k: int | None = None) -> float:
        r = recall[:, :, a, 0]
        if k is not None:
            r = r[:, k : k + 1]
        return _mean_valid(r)

    labels = list(bins.keys())
    a_all = labels.index("all")
    out = {
        "overall": {"AP": ap(a_all), "AP50": ap(a_all, t50), "AP75": ap(a_all, t75), "AR": ar(a_all)},
        "by_size": {
            lbl: {"AP": ap(a), "AP50": ap(a, t50), "AR": ar(a), "n_gt": n_gt[lbl]} for a, lbl in enumerate(labels)
        },
        "per_class": {
            names[cid]: {"AP": ap(a_all, None, k), "AP50": ap(a_all, t50, k), "AR": ar(a_all, k)}
            for k, cid in enumerate(cat_ids)
        },
        "per_class_by_size": {
            names[cid]: {lbl: ap(a, t50, k) for a, lbl in enumerate(labels)} for k, cid in enumerate(cat_ids)
        },
        "max_dets": max_dets,
        "n_predictions": len(preds),
    }
    return out


def match_predictions(
    gt_boxes_xyxy: np.ndarray,
    gt_cls: np.ndarray,
    det_xyxy: np.ndarray,
    det_cls: np.ndarray,
    det_conf: np.ndarray,
    iou_thr: float = 0.5,
) -> np.ndarray:
    """Greedy one-to-one matching of detections (by decreasing confidence) to same-class GT boxes.

    Returns an array ``gt_index`` of length ``len(det)``: the matched GT index or -1 (false positive).
    """
    from .tiling import iou_matrix

    gt_index = -np.ones(len(det_xyxy), dtype=np.int64)
    if len(gt_boxes_xyxy) == 0 or len(det_xyxy) == 0:
        return gt_index
    ious = iou_matrix(det_xyxy, gt_boxes_xyxy)
    ious[det_cls[:, None] != gt_cls[None, :]] = 0.0
    taken = np.zeros(len(gt_boxes_xyxy), dtype=bool)
    for d in np.argsort(-det_conf, kind="stable"):
        row = ious[d].copy()
        row[taken] = 0.0
        g = int(np.argmax(row))
        if row[g] >= iou_thr:
            gt_index[d] = g
            taken[g] = True
    return gt_index
