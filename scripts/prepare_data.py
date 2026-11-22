#!/usr/bin/env python3
"""Download VisDrone2019-DET splits, convert them to YOLO format and export COCO ground truth.

Usage: python scripts/prepare_data.py --splits train val [--raw data/raw] [--out data/visdrone]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from smallobj.visdrone import (
    convert_split,
    download_split,
    export_coco_gt,
    extract_split,
    stats_to_dict,
    write_data_yaml,
)

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", nargs="+", default=["train", "val"])
    ap.add_argument("--raw", type=Path, default=ROOT / "data" / "raw")
    ap.add_argument("--out", type=Path, default=ROOT / "data" / "visdrone")
    ap.add_argument("--copy-images", action="store_true", help="copy images instead of symlinking")
    args = ap.parse_args()

    metrics_dir = ROOT / "results" / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    for split in args.splits:
        zip_path = download_split(split, args.raw)
        split_dir = extract_split(zip_path, args.raw / "extracted")
        stats = convert_split(split_dir, args.out, split, link_images=not args.copy_images)
        gt = export_coco_gt(args.out, split, args.out / "annotations" / f"{split}_coco.json")
        d = stats_to_dict(stats)
        d["coco_annotations"] = len(gt["annotations"])
        (metrics_dir / f"dataset_{split}.json").write_text(json.dumps(d, indent=2))
        print(json.dumps(d, indent=2))
    write_data_yaml(args.out, args.out / "visdrone.yaml", tuple(args.splits))
    print(f"wrote {args.out / 'visdrone.yaml'}")


if __name__ == "__main__":
    main()
