"""Tiled ("sliced") inference helpers: tile grids, coordinate mapping and NMS merging.

Small objects seen from a drone are a few pixels wide once a 1360x765 frame is resized to
640 px. Tiling runs the detector on overlapping crops at native resolution and merges the
per-tile detections back into full-image coordinates. This is the idea behind SAHI
(Akyon et al., ICIP 2022); the implementation here is deliberately small and fully tested.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Tile:
    x0: int
    y0: int
    x1: int
    y1: int

    @property
    def width(self) -> int:
        return self.x1 - self.x0

    @property
    def height(self) -> int:
        return self.y1 - self.y0


def _starts(length: int, tile: int, stride: int) -> list[int]:
    """Start offsets along one axis so that tiles cover [0, length) with the last tile flush."""
    if length <= tile:
        return [0]
    starts = list(range(0, length - tile, stride))
    last = length - tile
    if starts[-1] != last:
        starts.append(last)
    return starts


def make_tiles(img_w: int, img_h: int, tile: int = 640, overlap: float = 0.2) -> list[Tile]:
    """Return a grid of overlapping tiles covering the image.

    ``overlap`` is the fraction of ``tile`` shared by neighbouring tiles. Tiles never exceed the
    image; if the image is smaller than ``tile`` on an axis, one tile spans the whole axis.
    """
    if not 0 <= overlap < 1:
        raise ValueError("overlap must be in [0, 1)")
    stride = max(1, int(round(tile * (1 - overlap))))
    tiles = []
    for y0 in _starts(img_h, tile, stride):
        for x0 in _starts(img_w, tile, stride):
            tiles.append(Tile(x0, y0, min(x0 + tile, img_w), min(y0 + tile, img_h)))
    return tiles


def crop(image: np.ndarray, tile: Tile) -> np.ndarray:
    return image[tile.y0 : tile.y1, tile.x0 : tile.x1]


def shift_boxes(boxes_xyxy: np.ndarray, tile: Tile) -> np.ndarray:
    """Map boxes predicted inside ``tile`` back to full-image coordinates."""
    boxes = np.asarray(boxes_xyxy, dtype=np.float64).reshape(-1, 4).copy()
    boxes[:, [0, 2]] += tile.x0
    boxes[:, [1, 3]] += tile.y0
    return boxes


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pairwise IoU between two (N,4) and (M,4) arrays of xyxy boxes."""
    a = np.asarray(a, dtype=np.float64).reshape(-1, 4)
    b = np.asarray(b, dtype=np.float64).reshape(-1, 4)
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    ix1 = np.maximum(a[:, None, 0], b[None, :, 0])
    iy1 = np.maximum(a[:, None, 1], b[None, :, 1])
    ix2 = np.minimum(a[:, None, 2], b[None, :, 2])
    iy2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(ix2 - ix1, 0, None) * np.clip(iy2 - iy1, 0, None)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    union = area_a[:, None] + area_b[None, :] - inter
    return np.where(union > 0, inter / np.maximum(union, 1e-12), 0.0)


def nms(boxes: np.ndarray, scores: np.ndarray, iou_thr: float = 0.5) -> np.ndarray:
    """Greedy non-maximum suppression. Returns indices to keep, sorted by decreasing score."""
    boxes = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    if len(boxes) == 0:
        return np.zeros(0, dtype=np.int64)
    order = np.argsort(-scores, kind="stable")
    keep: list[int] = []
    suppressed = np.zeros(len(boxes), dtype=bool)
    ious = iou_matrix(boxes, boxes)
    for i in order:
        if suppressed[i]:
            continue
        keep.append(int(i))
        suppressed |= ious[i] > iou_thr
        suppressed[i] = True
    return np.asarray(keep, dtype=np.int64)


def merge_detections(
    boxes: np.ndarray,
    scores: np.ndarray,
    classes: np.ndarray,
    iou_thr: float = 0.5,
    class_agnostic: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Merge detections gathered from several tiles (and optionally the full image) with NMS.

    Duplicates arise where tiles overlap: the same object is seen in two or more tiles. Per-class
    NMS (default) keeps the highest-scoring copy of each object.
    """
    boxes = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    classes = np.asarray(classes).reshape(-1)
    if len(boxes) == 0:
        return boxes, scores, classes
    if class_agnostic:
        keep = nms(boxes, scores, iou_thr)
    else:
        keep_list = []
        for c in np.unique(classes):
            idx = np.flatnonzero(classes == c)
            keep_list.append(idx[nms(boxes[idx], scores[idx], iou_thr)])
        keep = np.concatenate(keep_list) if keep_list else np.zeros(0, dtype=np.int64)
        keep = keep[np.argsort(-scores[keep], kind="stable")]
    return boxes[keep], scores[keep], classes[keep]
