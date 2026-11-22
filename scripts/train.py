#!/usr/bin/env python3
"""Fine-tune a YOLO11 model on VisDrone and record the run in results/metrics/train_<name>.json."""

from __future__ import annotations

import argparse
import json
import platform
import shutil
import time
from pathlib import Path

import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="yolo11n.pt", help="COCO-pretrained starting point")
    ap.add_argument("--weights", default=None, help="start from these weights instead (e.g. the 640 run)")
    ap.add_argument("--data", default=str(ROOT / "data" / "visdrone" / "visdrone.yaml"))
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--name", default="y11n_640")
    ap.add_argument("--patience", type=int, default=100)
    ap.add_argument("--fraction", type=float, default=1.0, help="fraction of the train set (smoke tests)")
    args = ap.parse_args()

    from ultralytics import YOLO

    model = YOLO(args.weights or args.model)
    t0 = time.time()
    model.train(
        data=args.data,
        imgsz=args.imgsz,
        epochs=args.epochs,
        batch=args.batch,
        workers=args.workers,
        device=args.device,
        seed=args.seed,
        project=str(ROOT / "runs"),
        name=args.name,
        exist_ok=True,
        patience=args.patience,
        fraction=args.fraction,
        plots=True,
        verbose=True,
    )
    train_seconds = time.time() - t0

    run_dir = ROOT / "runs" / args.name
    csv = pd.read_csv(run_dir / "results.csv")
    csv.columns = [c.strip() for c in csv.columns]
    best_idx = int(csv["metrics/mAP50-95(B)"].idxmax())
    best = csv.loc[best_idx]
    n_params = sum(p.numel() for p in model.model.parameters())
    record = {
        "name": args.name,
        "init": args.weights or args.model,
        "imgsz": args.imgsz,
        "epochs_requested": args.epochs,
        "epochs_run": int(len(csv)),
        "batch": args.batch,
        "seed": args.seed,
        "fraction": args.fraction,
        "device": args.device,
        "hardware": f"{platform.machine()} / torch {torch.__version__}",
        "train_seconds": round(train_seconds),
        "params": int(n_params),
        "best_epoch": int(best["epoch"]),
        "val_mAP50_95_ultralytics": float(best["metrics/mAP50-95(B)"]),
        "val_mAP50_ultralytics": float(best["metrics/mAP50(B)"]),
        "val_precision": float(best["metrics/precision(B)"]),
        "val_recall": float(best["metrics/recall(B)"]),
        "weights": str((run_dir / "weights" / "best.pt").relative_to(ROOT)),
    }
    out_dir = ROOT / "results" / "train" / args.name
    out_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(run_dir / "results.csv", out_dir / "results.csv")
    if (run_dir / "args.yaml").exists():
        # store paths relative to the project so the published copy carries no machine-specific prefix
        text = (run_dir / "args.yaml").read_text().replace(f"{ROOT}/", "")
        (out_dir / "args.yaml").write_text(text)
    for png in ("results.png", "confusion_matrix_normalized.png"):
        if (run_dir / png).exists():
            shutil.copy(run_dir / png, out_dir / png)
    (ROOT / "results" / "metrics").mkdir(parents=True, exist_ok=True)
    (ROOT / "results" / "metrics" / f"train_{args.name}.json").write_text(json.dumps(record, indent=2))
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
