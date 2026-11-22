#!/usr/bin/env python3
"""Evaluate trained models on VisDrone val: plain inference vs tiled inference, AP by size bin.

Each configuration writes results/predictions/<name>/{predictions.json,meta.json} and
results/metrics/eval_<name>.json. Configurations are listed in CONFIGS below; pass --only to
run a subset, --limit N to smoke-test on the first N images.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from smallobj.evaluate import evaluate_coco
from smallobj.infer import load_model, run_split
from smallobj.visdrone import load_ignored_regions

ROOT = Path(__file__).resolve().parents[1]

# name -> (weights run, mode, kwargs)
CONFIGS: dict[str, dict] = {
    "y11n_640_plain": {"run": "y11n_640", "mode": "plain", "imgsz": 640},
    "y11n_640_plain1024": {"run": "y11n_640", "mode": "plain", "imgsz": 1024},
    "y11n_640_tiled": {"run": "y11n_640", "mode": "tiled", "imgsz": 640, "tile": 640, "overlap": 0.2},
    "y11n_640_tiled_full": {
        "run": "y11n_640",
        "mode": "tiled",
        "imgsz": 640,
        "tile": 640,
        "overlap": 0.2,
        "include_full": True,
    },
    "y11n_1024_plain": {"run": "y11n_1024", "mode": "plain", "imgsz": 1024},
    "y11n_1024_tiled": {"run": "y11n_1024", "mode": "tiled", "imgsz": 1024, "tile": 1024, "overlap": 0.2},
    "y11n_1024_tiled640": {
        "run": "y11n_1024",
        "mode": "tiled",
        "imgsz": 1024,
        "tile": 640,
        "overlap": 0.2,
        "include_full": True,
    },
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", default=None, help="config names to run (default: all with weights)")
    ap.add_argument("--device", default="mps")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--data", type=Path, default=ROOT / "data" / "visdrone")
    ap.add_argument("--no-ignore-filter", action="store_true", help="keep predictions inside ignored regions")
    args = ap.parse_args()

    gt_json = args.data / "annotations" / "val_coco.json"
    coco_gt = json.loads(gt_json.read_text())
    ignored = None if args.no_ignore_filter else load_ignored_regions(args.data, "val")
    names = args.only or list(CONFIGS)
    models: dict[str, object] = {}
    for name in names:
        cfg = CONFIGS[name]
        weights = ROOT / "runs" / cfg["run"] / "weights" / "best.pt"
        if not weights.exists():
            print(f"skip {name}: {weights} not found")
            continue
        if cfg["run"] not in models:
            models[cfg["run"]] = load_model(weights)
        out_dir = ROOT / "results" / "predictions" / name
        meta = run_split(
            models[cfg["run"]],
            coco_gt,
            args.data / "images" / "val",
            cfg["mode"],
            out_dir,
            ignored=ignored,
            imgsz=cfg["imgsz"],
            tile=cfg.get("tile", 640),
            overlap=cfg.get("overlap", 0.2),
            include_full=cfg.get("include_full", False),
            device=args.device,
            limit=args.limit,
        )
        gt_for_eval = gt_json
        if args.limit:  # restrict GT to the evaluated images
            sub = dict(coco_gt)
            ids = {im["id"] for im in coco_gt["images"][: args.limit]}
            sub["images"] = [im for im in coco_gt["images"] if im["id"] in ids]
            sub["annotations"] = [a for a in coco_gt["annotations"] if a["image_id"] in ids]
            gt_for_eval = out_dir / "gt_subset.json"
            gt_for_eval.write_text(json.dumps(sub))
        metrics = evaluate_coco(gt_for_eval, out_dir / "predictions.json")
        metrics["config"] = {"name": name, **cfg, **meta}
        (ROOT / "results" / "metrics").mkdir(parents=True, exist_ok=True)
        (ROOT / "results" / "metrics" / f"eval_{name}.json").write_text(json.dumps(metrics, indent=2))
        o, s = metrics["overall"], metrics["by_size"]
        print(
            f"{name:22s} AP={o['AP']:.3f} AP50={o['AP50']:.3f} | tiny={s['tiny']['AP50']:.3f} "
            f"small={s['small']['AP50']:.3f} medium={s['medium']['AP50']:.3f} "
            f"large={s['large']['AP50']:.3f} | {meta['seconds_per_image']:.2f}s/img"
        )


if __name__ == "__main__":
    main()
