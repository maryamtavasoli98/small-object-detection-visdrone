"""VisDrone2019-DET: download, parse, convert to YOLO format and export COCO ground truth.

Annotation format of VisDrone-DET (one txt per image, one object per line):

    <bbox_left>,<bbox_top>,<bbox_width>,<bbox_height>,<score>,<object_category>,<truncation>,<occlusion>

* ``score`` is 1 for objects to evaluate and 0 for *ignored regions*.
* ``object_category``: 0 ignored regions, 1 pedestrian, 2 people, 3 bicycle, 4 car, 5 van,
  6 truck, 7 tricycle, 8 awning-tricycle, 9 bus, 10 motor, 11 others.
* ``truncation``: 0 (no truncation) or 1 (1-50 % of the object outside the frame).
* ``occlusion``: 0 (none), 1 (partial, 1-50 %), 2 (heavy, 50-100 %).

We keep categories 1-10 (ten classes), drop "others" (11) and store the ignored regions
separately so that predictions falling inside them can be discarded at evaluation time,
as the official VisDrone toolkit does.
"""

from __future__ import annotations

import csv
import json
import os
import urllib.request
import zipfile
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path

from PIL import Image

CLASSES: list[str] = [
    "pedestrian",
    "people",
    "bicycle",
    "car",
    "van",
    "truck",
    "tricycle",
    "awning-tricycle",
    "bus",
    "motor",
]
IGNORED_CATEGORY = 0
OTHERS_CATEGORY = 11

RELEASE = "https://github.com/ultralytics/assets/releases/download/v0.0.0"
URLS: dict[str, str] = {
    "train": f"{RELEASE}/VisDrone2019-DET-train.zip",
    "val": f"{RELEASE}/VisDrone2019-DET-val.zip",
    "test-dev": f"{RELEASE}/VisDrone2019-DET-test-dev.zip",
}


@dataclass
class RawBox:
    """One annotation line, in pixels of the original image."""

    left: float
    top: float
    width: float
    height: float
    score: int
    category: int
    truncation: int
    occlusion: int

    @property
    def is_ignored_region(self) -> bool:
        return self.score == 0 or self.category == IGNORED_CATEGORY

    @property
    def area(self) -> float:
        return self.width * self.height


@dataclass
class ConversionStats:
    split: str
    images: int = 0
    kept_boxes: int = 0
    ignored_regions: int = 0
    dropped_others: int = 0
    dropped_degenerate: int = 0
    per_class: dict[str, int] = field(default_factory=dict)


def parse_annotation_file(path: Path) -> list[RawBox]:
    """Parse one VisDrone annotation file. Lines may end with a trailing comma."""
    boxes: list[RawBox] = []
    for line in path.read_text().splitlines():
        line = line.strip().rstrip(",")
        if not line:
            continue
        parts = line.split(",")
        if len(parts) < 8:
            raise ValueError(f"{path}: expected 8 fields, got {len(parts)}: {line!r}")
        left, top, width, height = (float(parts[i]) for i in range(4))
        score, category, truncation, occlusion = (int(float(parts[i])) for i in range(4, 8))
        boxes.append(RawBox(left, top, width, height, score, category, truncation, occlusion))
    return boxes


def download_split(split: str, raw_dir: Path) -> Path:
    """Download one split archive into ``raw_dir`` if missing. Returns the zip path."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    zip_path = raw_dir / f"VisDrone2019-DET-{split}.zip"
    if zip_path.exists() and zip_path.stat().st_size > 0:
        return zip_path
    tmp = zip_path.with_suffix(".zip.part")
    print(f"downloading {URLS[split]} -> {zip_path}")
    urllib.request.urlretrieve(URLS[split], tmp)  # noqa: S310 (fixed https URLs)
    tmp.rename(zip_path)
    return zip_path


def extract_split(zip_path: Path, extract_dir: Path) -> Path:
    """Extract the archive (idempotent). Returns the split folder with images/ and annotations/."""
    target = extract_dir / zip_path.stem
    if (target / "images").is_dir() and (target / "annotations").is_dir():
        return target
    extract_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(extract_dir)
    if not (target / "images").is_dir():
        raise FileNotFoundError(f"expected {target}/images after extracting {zip_path}")
    return target


def _clip_box(left: float, top: float, width: float, height: float, w: int, h: int):
    x1 = min(max(left, 0.0), w)
    y1 = min(max(top, 0.0), h)
    x2 = min(max(left + width, 0.0), w)
    y2 = min(max(top + height, 0.0), h)
    return x1, y1, x2, y2


def convert_split(split_dir: Path, out_dir: Path, split: str, link_images: bool = True) -> ConversionStats:
    """Convert one extracted split into the Ultralytics layout.

    Writes
      out_dir/images/<split>/<name>.jpg      (symlink to the raw image, or a copy)
      out_dir/labels/<split>/<name>.txt      (YOLO: cls cx cy w h, normalised)
      out_dir/ignored/<split>/<name>.txt     (ignored regions: x1 y1 x2 y2 in pixels)
      out_dir/meta/<split>_boxes.csv         (every kept box with size/occlusion/truncation)
    """
    img_out = out_dir / "images" / split
    lbl_out = out_dir / "labels" / split
    ign_out = out_dir / "ignored" / split
    meta_out = out_dir / "meta"
    for d in (img_out, lbl_out, ign_out, meta_out):
        d.mkdir(parents=True, exist_ok=True)

    stats = ConversionStats(split=split)
    counter: Counter[str] = Counter()
    csv_path = meta_out / f"{split}_boxes.csv"
    with csv_path.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(
            [
                "image",
                "img_w",
                "img_h",
                "cls",
                "name",
                "x1",
                "y1",
                "x2",
                "y2",
                "w",
                "h",
                "area",
                "truncation",
                "occlusion",
            ]
        )
        for ann_path in sorted((split_dir / "annotations").glob("*.txt")):
            stem = ann_path.stem
            img_path = split_dir / "images" / f"{stem}.jpg"
            if not img_path.exists():
                raise FileNotFoundError(img_path)
            with Image.open(img_path) as im:
                w, h = im.size
            stats.images += 1

            dst = img_out / img_path.name
            if not dst.exists():
                if link_images:
                    os.symlink(img_path.resolve(), dst)
                else:
                    dst.write_bytes(img_path.read_bytes())

            label_lines: list[str] = []
            ignored_lines: list[str] = []
            for b in parse_annotation_file(ann_path):
                x1, y1, x2, y2 = _clip_box(b.left, b.top, b.width, b.height, w, h)
                if b.is_ignored_region:
                    stats.ignored_regions += 1
                    ignored_lines.append(f"{x1:.1f} {y1:.1f} {x2:.1f} {y2:.1f}")
                    continue
                if b.category == OTHERS_CATEGORY or not 1 <= b.category <= len(CLASSES):
                    stats.dropped_others += 1
                    continue
                bw, bh = x2 - x1, y2 - y1
                if bw < 1 or bh < 1:
                    stats.dropped_degenerate += 1
                    continue
                cls = b.category - 1
                name = CLASSES[cls]
                counter[name] += 1
                stats.kept_boxes += 1
                cx, cy = (x1 + x2) / 2 / w, (y1 + y2) / 2 / h
                label_lines.append(f"{cls} {cx:.6f} {cy:.6f} {bw / w:.6f} {bh / h:.6f}")
                writer.writerow(
                    [
                        stem,
                        w,
                        h,
                        cls,
                        name,
                        f"{x1:.1f}",
                        f"{y1:.1f}",
                        f"{x2:.1f}",
                        f"{y2:.1f}",
                        f"{bw:.1f}",
                        f"{bh:.1f}",
                        f"{bw * bh:.1f}",
                        b.truncation,
                        b.occlusion,
                    ]
                )
            (lbl_out / f"{stem}.txt").write_text("\n".join(label_lines) + ("\n" if label_lines else ""))
            (ign_out / f"{stem}.txt").write_text("\n".join(ignored_lines) + ("\n" if ignored_lines else ""))

    stats.per_class = dict(sorted(counter.items(), key=lambda kv: -kv[1]))
    return stats


def load_ignored_regions(out_dir: Path, split: str) -> dict[str, list[tuple[float, float, float, float]]]:
    """Return {image_stem: [(x1, y1, x2, y2), ...]} of ignored regions for a converted split."""
    regions: dict[str, list[tuple[float, float, float, float]]] = {}
    for p in sorted((out_dir / "ignored" / split).glob("*.txt")):
        boxes = []
        for line in p.read_text().splitlines():
            if line.strip():
                x1, y1, x2, y2 = (float(v) for v in line.split())
                boxes.append((x1, y1, x2, y2))
        regions[p.stem] = boxes
    return regions


def export_coco_gt(out_dir: Path, split: str, json_path: Path) -> dict:
    """Write a COCO-format ground-truth file for a converted split (from the YOLO labels)."""
    images, annotations = [], []
    ann_id = 1
    lbl_dir = out_dir / "labels" / split
    img_dir = out_dir / "images" / split
    for image_id, lbl_path in enumerate(sorted(lbl_dir.glob("*.txt")), start=1):
        stem = lbl_path.stem
        with Image.open(img_dir / f"{stem}.jpg") as im:
            w, h = im.size
        images.append({"id": image_id, "file_name": f"{stem}.jpg", "width": w, "height": h})
        for line in lbl_path.read_text().splitlines():
            if not line.strip():
                continue
            cls, cx, cy, bw, bh = line.split()
            bw_px, bh_px = float(bw) * w, float(bh) * h
            x1 = float(cx) * w - bw_px / 2
            y1 = float(cy) * h - bh_px / 2
            annotations.append(
                {
                    "id": ann_id,
                    "image_id": image_id,
                    "category_id": int(cls) + 1,
                    "bbox": [round(x1, 2), round(y1, 2), round(bw_px, 2), round(bh_px, 2)],
                    "area": round(bw_px * bh_px, 2),
                    "iscrowd": 0,
                }
            )
            ann_id += 1
    coco = {
        "info": {"description": f"VisDrone2019-DET {split} (converted by smallobj)"},
        "images": images,
        "annotations": annotations,
        "categories": [{"id": i + 1, "name": n} for i, n in enumerate(CLASSES)],
    }
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(coco))
    return coco


def write_data_yaml(out_dir: Path, yaml_path: Path, splits: tuple[str, ...] = ("train", "val")) -> None:
    """Write the Ultralytics dataset yaml pointing at the converted folders."""
    lines = [f"path: {out_dir.resolve()}"]
    for s in splits:
        key = "val" if s == "val" else ("test" if s == "test-dev" else s)
        lines.append(f"{key}: images/{s}")
    lines.append("names:")
    lines += [f"  {i}: {n}" for i, n in enumerate(CLASSES)]
    yaml_path.parent.mkdir(parents=True, exist_ok=True)
    yaml_path.write_text("\n".join(lines) + "\n")


def stats_to_dict(stats: ConversionStats) -> dict:
    return asdict(stats)
