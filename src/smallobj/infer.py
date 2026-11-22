"""Plain and tiled inference with an Ultralytics YOLO model, producing COCO-style results."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .tiling import crop, make_tiles, merge_detections, shift_boxes


@dataclass
class Detections:
    xyxy: np.ndarray  # (N, 4) float, full-image pixel coordinates
    conf: np.ndarray  # (N,) float
    cls: np.ndarray  # (N,) int

    @classmethod
    def empty(cls) -> Detections:
        return cls(np.zeros((0, 4)), np.zeros(0), np.zeros(0, dtype=np.int64))

    @classmethod
    def concat(cls, parts: list[Detections]) -> Detections:
        parts = [p for p in parts if len(p)]
        if not parts:
            return cls.empty()
        return cls(
            np.concatenate([p.xyxy for p in parts]),
            np.concatenate([p.conf for p in parts]),
            np.concatenate([p.cls for p in parts]),
        )

    def __len__(self) -> int:
        return int(len(self.conf))

    def filter_ignored(self, regions: list[tuple[float, float, float, float]]) -> Detections:
        """Drop detections whose centre lies inside an ignored region (VisDrone protocol)."""
        if not regions or len(self) == 0:
            return self
        cx = (self.xyxy[:, 0] + self.xyxy[:, 2]) / 2
        cy = (self.xyxy[:, 1] + self.xyxy[:, 3]) / 2
        inside = np.zeros(len(self), dtype=bool)
        for x1, y1, x2, y2 in regions:
            inside |= (cx >= x1) & (cx <= x2) & (cy >= y1) & (cy <= y2)
        keep = ~inside
        return Detections(self.xyxy[keep], self.conf[keep], self.cls[keep])

    def to_coco(self, image_id: int) -> list[dict]:
        out = []
        for (x1, y1, x2, y2), s, c in zip(self.xyxy, self.conf, self.cls, strict=True):
            out.append(
                {
                    "image_id": int(image_id),
                    "category_id": int(c) + 1,
                    "bbox": [
                        round(float(x1), 2),
                        round(float(y1), 2),
                        round(float(x2 - x1), 2),
                        round(float(y2 - y1), 2),
                    ],
                    "score": round(float(s), 5),
                }
            )
        return out


def load_model(weights: str | Path):
    from ultralytics import YOLO

    return YOLO(str(weights))


def results_to_detections(result) -> Detections:
    """Convert one Ultralytics ``Results`` object to :class:`Detections`."""
    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        return Detections.empty()
    return Detections(
        boxes.xyxy.cpu().numpy().astype(np.float64),
        boxes.conf.cpu().numpy().astype(np.float64),
        boxes.cls.cpu().numpy().astype(np.int64),
    )


def predict_batch(
    model, images: list[np.ndarray], imgsz: int, conf: float, iou: float, max_det: int, device
) -> list[Detections]:
    results = model.predict(images, imgsz=imgsz, conf=conf, iou=iou, max_det=max_det, device=device, verbose=False)
    return [results_to_detections(r) for r in results]


def predict_full(
    model, image: np.ndarray, imgsz: int = 640, conf: float = 0.001, iou: float = 0.6, max_det: int = 1000, device=None
) -> Detections:
    """Run the detector once on the whole (resized, letterboxed) image."""
    return predict_batch(model, [image], imgsz, conf, iou, max_det, device)[0]


def predict_tiled(
    model,
    image: np.ndarray,
    tile: int = 640,
    overlap: float = 0.2,
    conf: float = 0.001,
    iou: float = 0.6,
    max_det_per_tile: int = 300,
    merge_iou: float = 0.5,
    include_full: bool = False,
    full_imgsz: int | None = None,
    device=None,
    batch: int = 8,
) -> Detections:
    """Run the detector on overlapping native-resolution tiles and merge with per-class NMS.

    With ``include_full`` the resized full image is added as an extra "tile", which helps for
    large objects that no single tile contains completely.
    """
    h, w = image.shape[:2]
    tiles = make_tiles(w, h, tile, overlap)
    crops = [crop(image, t) for t in tiles]
    parts: list[Detections] = []
    for i in range(0, len(crops), batch):
        dets = predict_batch(model, crops[i : i + batch], tile, conf, iou, max_det_per_tile, device)
        for d, t in zip(dets, tiles[i : i + batch], strict=True):
            if len(d):
                parts.append(Detections(shift_boxes(d.xyxy, t), d.conf, d.cls))
    if include_full:
        parts.append(predict_full(model, image, full_imgsz or tile, conf, iou, 1000, device))
    allv = Detections.concat(parts)
    if len(allv) == 0:
        return allv
    boxes, scores, classes = merge_detections(allv.xyxy, allv.conf, allv.cls, iou_thr=merge_iou)
    return Detections(boxes, scores, classes)


def run_split(
    model,
    coco_gt: dict,
    image_dir: Path,
    mode: str,
    out_dir: Path,
    ignored: dict[str, list[tuple[float, float, float, float]]] | None = None,
    imgsz: int = 640,
    tile: int = 640,
    overlap: float = 0.2,
    include_full: bool = False,
    conf: float = 0.001,
    iou: float = 0.6,
    device=None,
    limit: int | None = None,
) -> dict:
    """Predict every image of a split and write ``predictions.json`` (COCO results) + ``meta.json``."""
    out_dir.mkdir(parents=True, exist_ok=True)
    preds: list[dict] = []
    times: list[float] = []
    n_tiles = 0
    images = coco_gt["images"][:limit] if limit else coco_gt["images"]
    for info in images:
        image = cv2.imread(str(image_dir / info["file_name"]))
        if image is None:
            raise FileNotFoundError(image_dir / info["file_name"])
        t0 = time.perf_counter()
        if mode == "plain":
            det = predict_full(model, image, imgsz, conf, iou, 1000, device)
        elif mode == "tiled":
            det = predict_tiled(model, image, tile, overlap, conf, iou, 300, 0.5, include_full, imgsz, device)
            n_tiles += len(make_tiles(image.shape[1], image.shape[0], tile, overlap)) + int(include_full)
        else:
            raise ValueError(mode)
        times.append(time.perf_counter() - t0)
        stem = Path(info["file_name"]).stem
        if ignored:
            det = det.filter_ignored(ignored.get(stem, []))
        preds.extend(det.to_coco(info["id"]))
    meta = {
        "mode": mode,
        "imgsz": imgsz,
        "tile": tile if mode == "tiled" else None,
        "overlap": overlap if mode == "tiled" else None,
        "include_full": include_full if mode == "tiled" else None,
        "conf": conf,
        "nms_iou": iou,
        "images": len(images),
        "detections": len(preds),
        "tiles_per_image": (n_tiles / len(images)) if mode == "tiled" and images else None,
        "seconds_per_image": float(np.mean(times)) if times else None,
        "device": str(device),
    }
    (out_dir / "predictions.json").write_text(json.dumps(preds))
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    return meta
