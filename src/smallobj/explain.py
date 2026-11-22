"""D-RISE: black-box saliency for object detectors, plus deletion/insertion faithfulness curves.

D-RISE (Petsiuk et al., CVPR 2021) extends RISE to detection: thousands of random binary masks
are applied to the image, the detector is run on each masked image, and every mask is weighted
by how well the detector still finds the *target* detection (same class, high IoU, high score).
The saliency map is the weighted average of the masks. It needs no access to gradients or to
the model's internals, so it works for any detector, including Ultralytics YOLO.

Faithfulness is measured with deletion and insertion curves (Petsiuk et al., BMVC 2018): remove
(or insert) pixels in order of decreasing saliency and track the target's detection score. A
faithful map makes the score drop fast under deletion (low AUC) and rise fast under insertion
(high AUC). A random pixel order gives the reference.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np

from .infer import Detections, predict_batch
from .tiling import iou_matrix


def generate_masks(
    n: int, hw: tuple[int, int], grid: int = 16, p: float = 0.5, rng: np.random.Generator | None = None
) -> np.ndarray:
    """RISE-style masks: a coarse ``grid x grid`` Bernoulli(p) pattern, bilinearly upsampled and
    randomly shifted by up to one cell. Returns float32 array of shape (n, H, W) in [0, 1]."""
    rng = rng or np.random.default_rng(0)
    h, w = hw
    cell_h, cell_w = math.ceil(h / grid), math.ceil(w / grid)
    up_h, up_w = (grid + 1) * cell_h, (grid + 1) * cell_w
    masks = np.empty((n, h, w), dtype=np.float32)
    for i in range(n):
        coarse = (rng.random((grid, grid)) < p).astype(np.float32)
        up = cv2.resize(coarse, (up_w, up_h), interpolation=cv2.INTER_LINEAR)
        dy, dx = int(rng.integers(0, cell_h)), int(rng.integers(0, cell_w))
        masks[i] = up[dy : dy + h, dx : dx + w]
    return masks


def apply_mask(image: np.ndarray, mask: np.ndarray, fill: np.ndarray | int = 0) -> np.ndarray:
    """Blend ``image`` with ``fill`` (scalar or same-shape image) according to ``mask`` in [0,1]."""
    m = mask[..., None].astype(np.float32)
    fill_img = (
        np.broadcast_to(np.asarray(fill, dtype=np.float32), image.shape)
        if np.ndim(fill) == 0
        else fill.astype(np.float32)
    )
    return (image.astype(np.float32) * m + fill_img * (1.0 - m)).clip(0, 255).astype(np.uint8)


def similarity(target_xyxy: np.ndarray, target_cls: int, det: Detections) -> float:
    """D-RISE similarity between the target and the best matching detection: IoU x class score.

    Ultralytics exposes one class and one confidence per box, so we use an indicator on the class
    instead of the cosine similarity between class-probability vectors used in the paper.
    """
    if len(det) == 0:
        return 0.0
    same = det.cls == target_cls
    if not same.any():
        return 0.0
    ious = iou_matrix(np.asarray(target_xyxy)[None, :], det.xyxy[same])[0]
    return float(np.max(ious * det.conf[same]))


@dataclass
class DriseResult:
    saliency: np.ndarray  # (H, W) float32, normalised to [0, 1]
    base_score: float  # similarity on the unmasked image
    n_masks: int


def drise(
    model,
    image: np.ndarray,
    target_xyxy: np.ndarray,
    target_cls: int,
    imgsz: int,
    n_masks: int = 1000,
    grid: int = 16,
    p: float = 0.5,
    batch: int = 16,
    conf: float = 0.01,
    device=None,
    rng: np.random.Generator | None = None,
) -> DriseResult:
    """Compute the D-RISE saliency map of one target detection in ``image`` (BGR, uint8)."""
    rng = rng or np.random.default_rng(0)
    h, w = image.shape[:2]
    base = similarity(target_xyxy, target_cls, predict_batch(model, [image], imgsz, conf, 0.6, 300, device)[0])
    sal = np.zeros((h, w), dtype=np.float64)
    done = 0
    while done < n_masks:
        k = min(batch, n_masks - done)
        masks = generate_masks(k, (h, w), grid, p, rng)
        dets = predict_batch(model, [apply_mask(image, m) for m in masks], imgsz, conf, 0.6, 300, device)
        for m, d in zip(masks, dets, strict=True):
            sal += similarity(target_xyxy, target_cls, d) * m
        done += k
    sal /= max(n_masks * p, 1e-12)
    if sal.max() > 0:
        sal = sal / sal.max()
    return DriseResult(sal.astype(np.float32), base, n_masks)


def pixel_order(saliency: np.ndarray, how: str, rng: np.random.Generator | None = None) -> np.ndarray:
    """Flat pixel indices ordered for deletion/insertion: by saliency, or random."""
    if how == "saliency":
        return np.argsort(-saliency.ravel(), kind="stable")
    if how == "random":
        rng = rng or np.random.default_rng(0)
        return rng.permutation(saliency.size)
    raise ValueError(how)


def _blurred(image: np.ndarray, ksize: int = 51) -> np.ndarray:
    return cv2.GaussianBlur(image, (ksize, ksize), 0)


def deletion_insertion(
    model,
    image: np.ndarray,
    saliency: np.ndarray,
    target_xyxy: np.ndarray,
    target_cls: int,
    imgsz: int,
    order: str = "saliency",
    steps: int = 20,
    batch: int = 16,
    conf: float = 0.01,
    device=None,
    rng: np.random.Generator | None = None,
) -> dict:
    """Deletion and insertion curves for one target. Returns fractions, scores and AUCs.

    Deletion replaces pixels (in ``order``) of the original image by a heavily blurred copy;
    insertion starts from the blurred copy and restores pixels. Scores are the D-RISE
    similarity of the target, normalised by the unmasked score so curves start/end near 1.
    """
    h, w = image.shape[:2]
    idx = pixel_order(saliency, order, rng)
    fractions = np.linspace(0.0, 1.0, steps + 1)
    blurred = _blurred(image)
    del_imgs, ins_imgs = [], []
    for f in fractions:
        k = int(round(f * idx.size))
        mask = np.ones(h * w, dtype=np.float32)
        mask[idx[:k]] = 0.0
        mask = mask.reshape(h, w)
        del_imgs.append(apply_mask(image, mask, blurred))  # keep where mask=1, blur where removed
        ins_imgs.append(apply_mask(image, 1.0 - mask, blurred))  # restore the removed set only
    base = similarity(target_xyxy, target_cls, predict_batch(model, [image], imgsz, conf, 0.6, 300, device)[0])
    base = max(base, 1e-6)

    def scores(imgs: list[np.ndarray]) -> np.ndarray:
        out = []
        for i in range(0, len(imgs), batch):
            out += [
                similarity(target_xyxy, target_cls, d)
                for d in predict_batch(model, imgs[i : i + batch], imgsz, conf, 0.6, 300, device)
            ]
        return np.asarray(out) / base

    d, ins = scores(del_imgs), scores(ins_imgs)
    return {
        "order": order,
        "fractions": fractions.tolist(),
        "deletion": d.tolist(),
        "insertion": ins.tolist(),
        "deletion_auc": float(np.trapezoid(d, fractions)),
        "insertion_auc": float(np.trapezoid(ins, fractions)),
    }


def saliency_stats(saliency: np.ndarray, target_xyxy: np.ndarray, quantile: float = 0.9) -> dict:
    """How concentrated is the map? Mass inside the target box, and the image fraction needed
    to hold ``quantile`` of the total saliency mass (smaller = more focused)."""
    s = saliency.astype(np.float64)
    total = s.sum()
    if total <= 0:
        return {"mass_in_box": 0.0, "area_frac_90": 1.0, "box_area_frac": 0.0}
    x1, y1, x2, y2 = (int(round(v)) for v in target_xyxy)
    h, w = s.shape
    x1, y1, x2, y2 = max(x1, 0), max(y1, 0), min(x2, w), min(y2, h)
    inside = s[y1:y2, x1:x2].sum() / total
    flat = np.sort(s.ravel())[::-1]
    cum = np.cumsum(flat) / total
    k = int(np.searchsorted(cum, quantile)) + 1
    return {
        "mass_in_box": float(inside),
        "area_frac_90": float(k / s.size),
        "box_area_frac": float(((x2 - x1) * (y2 - y1)) / s.size),
    }
