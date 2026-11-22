#!/usr/bin/env python3
"""Result figures from results/metrics/*.json (+ qualitative plain-vs-tiled detections)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from smallobj.visdrone import CLASSES  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
METRICS = ROOT / "results" / "metrics"
FIGS = ROOT / "results" / "figures"
BIN_ORDER = ["tiny", "small", "medium", "large", "all"]
BIN_LABEL = {
    "tiny": "tiny (<16 px)",
    "small": "small (16-32)",
    "medium": "medium (32-96)",
    "large": "large (>96)",
    "all": "all",
}


def load_evals() -> dict[str, dict]:
    return {p.stem.removeprefix("eval_"): json.loads(p.read_text()) for p in sorted(METRICS.glob("eval_*.json"))}


def pretty(name: str) -> str:
    return {
        "y11n_640_plain": "640 model, plain 640",
        "y11n_640_plain1024": "640 model, plain 1024",
        "y11n_640_tiled": "640 model, 640 tiles",
        "y11n_640_tiled_full": "640 model, 640 tiles + full",
        "y11n_1024_plain": "1024 model, plain 1024",
        "y11n_1024_tiled": "1024 model, 1024 tiles",
        "y11n_1024_tiled640": "1024 model, 640 tiles + full",
    }.get(name, name)


def fig_ap_by_size(evals: dict[str, dict]) -> None:
    names = list(evals)
    bins = [b for b in BIN_ORDER if b != "all"]
    fig, ax = plt.subplots(figsize=(8.5, 3.8))
    width = 0.8 / max(len(names), 1)
    for i, n in enumerate(names):
        vals = [evals[n]["by_size"][b]["AP50"] for b in bins]
        ax.bar(np.arange(len(bins)) + i * width, vals, width, label=pretty(n))
    ax.set_xticks(np.arange(len(bins)) + width * (len(names) - 1) / 2)
    ax.set_xticklabels([BIN_LABEL[b] for b in bins])
    ax.set_ylabel("AP50 (val)")
    ax.set_title("Where tiling helps: AP50 by object size (original-image pixels)")
    ax.legend(fontsize=7.5, frameon=False, ncol=2)
    fig.tight_layout()
    fig.savefig(FIGS / "ap_by_size.png", dpi=160)
    plt.close(fig)


def fig_per_class(evals: dict[str, dict], a: str = "y11n_640_plain", b: str = "y11n_640_tiled") -> None:
    if a not in evals or b not in evals:
        return
    fig, ax = plt.subplots(figsize=(8.5, 3.6))
    x = np.arange(len(CLASSES))
    va = [evals[a]["per_class"][c]["AP50"] for c in CLASSES]
    vb = [evals[b]["per_class"][c]["AP50"] for c in CLASSES]
    ax.bar(x - 0.2, va, 0.4, label=pretty(a))
    ax.bar(x + 0.2, vb, 0.4, label=pretty(b))
    ax.set_xticks(x)
    ax.set_xticklabels(CLASSES, rotation=25, ha="right", fontsize=8)
    ax.set_ylabel("AP50 (val)")
    ax.set_title("Per-class AP50: plain vs tiled inference (same weights)")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGS / "per_class_ap50.png", dpi=160)
    plt.close(fig)


def fig_training_curves() -> None:
    runs = sorted((ROOT / "results" / "train").glob("*/results.csv"))
    if not runs:
        return
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.2))
    for csv in runs:
        df = pd.read_csv(csv)
        df.columns = [c.strip() for c in df.columns]
        axes[0].plot(df["epoch"], df["metrics/mAP50(B)"], label=csv.parent.name)
        axes[1].plot(df["epoch"], df["train/box_loss"], label=f"{csv.parent.name} train")
        axes[1].plot(df["epoch"], df["val/box_loss"], "--", label=f"{csv.parent.name} val")
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("val mAP50 (Ultralytics)")
    axes[0].legend(frameon=False, fontsize=8)
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("box loss")
    axes[1].legend(frameon=False, fontsize=7)
    fig.tight_layout()
    fig.savefig(FIGS / "training_curves.png", dpi=160)
    plt.close(fig)


def fig_explain_summary() -> None:
    p = METRICS / "explain_drise.json"
    if not p.exists():
        return
    data = json.loads(p.read_text())
    bins = [b for b in ["tiny", "small", "medium", "large"] if b in data["summary"]]
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.4))
    x = np.arange(len(bins))
    s = data["summary"]
    axes[0].bar(
        x - 0.2,
        [s[b]["deletion_auc_drise"]["mean"] for b in bins],
        0.4,
        yerr=[s[b]["deletion_auc_drise"]["std"] for b in bins],
        label="D-RISE order",
        color="#b2182b",
        capsize=2,
    )
    axes[0].bar(
        x + 0.2,
        [s[b]["deletion_auc_random"]["mean"] for b in bins],
        0.4,
        yerr=[s[b]["deletion_auc_random"]["std"] for b in bins],
        label="random order",
        color="#bbbbbb",
        capsize=2,
    )
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([BIN_LABEL[b] for b in bins], fontsize=8)
    axes[0].set_ylabel("deletion AUC  (lower = more faithful)")
    axes[0].legend(frameon=False, fontsize=8)
    axes[1].bar(
        x - 0.2,
        [s[b]["insertion_auc_drise"]["mean"] for b in bins],
        0.4,
        yerr=[s[b]["insertion_auc_drise"]["std"] for b in bins],
        label="D-RISE order",
        color="#2166ac",
        capsize=2,
    )
    axes[1].bar(
        x + 0.2,
        [s[b]["insertion_auc_random"]["mean"] for b in bins],
        0.4,
        yerr=[s[b]["insertion_auc_random"]["std"] for b in bins],
        label="random order",
        color="#bbbbbb",
        capsize=2,
    )
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([BIN_LABEL[b] for b in bins], fontsize=8)
    axes[1].set_ylabel("insertion AUC  (higher = more faithful)")
    axes[1].legend(frameon=False, fontsize=8)
    fig.suptitle("Faithfulness of D-RISE saliency vs a random pixel order, by object size", fontsize=10)
    fig.tight_layout()
    fig.savefig(FIGS / "explain_faithfulness.png", dpi=160)
    plt.close(fig)


def fig_qualitative(weights: Path, data: Path, device: str, n_images: int = 2) -> None:
    """Pick the val images with most tiny GT objects; draw GT vs plain-640 vs tiled detections."""
    from smallobj.infer import load_model, predict_full, predict_tiled

    gt = json.loads((data / "annotations" / "val_coco.json").read_text())
    per_image: dict[int, list] = {}
    for a in gt["annotations"]:
        per_image.setdefault(a["image_id"], []).append(a)
    tiny_counts = {iid: sum(np.sqrt(a["area"]) < 16 for a in anns) for iid, anns in per_image.items()}
    chosen = sorted(tiny_counts, key=lambda k: -tiny_counts[k])[:n_images]
    model = load_model(weights)
    info = {im["id"]: im for im in gt["images"]}
    fig, axes = plt.subplots(n_images, 3, figsize=(15, 4.6 * n_images))
    axes = np.atleast_2d(axes)
    for row, iid in enumerate(chosen):
        img = cv2.imread(str(data / "images" / "val" / info[iid]["file_name"]))
        anns = per_image[iid]
        # crop a 700x450 window around the densest cluster of tiny objects
        tiny = np.array(
            [
                [a["bbox"][0] + a["bbox"][2] / 2, a["bbox"][1] + a["bbox"][3] / 2]
                for a in anns
                if np.sqrt(a["area"]) < 16
            ]
        )
        cx, cy = np.median(tiny, axis=0) if len(tiny) else (img.shape[1] / 2, img.shape[0] / 2)
        x0 = int(np.clip(cx - 350, 0, max(img.shape[1] - 700, 0)))
        y0 = int(np.clip(cy - 225, 0, max(img.shape[0] - 450, 0)))
        x1, y1 = min(x0 + 700, img.shape[1]), min(y0 + 450, img.shape[0])
        plain = predict_full(model, img, 640, 0.25, 0.6, 1000, device)
        tiled = predict_tiled(model, img, 640, 0.2, 0.25, 0.6, 300, 0.5, False, 640, device)
        rgb = cv2.cvtColor(img[y0:y1, x0:x1], cv2.COLOR_BGR2RGB)
        panels = [
            (
                "ground truth",
                [(a["bbox"][0], a["bbox"][1], a["bbox"][0] + a["bbox"][2], a["bbox"][1] + a["bbox"][3]) for a in anns],
                "lime",
            ),
            (f"plain 640 ({len(plain)} boxes @conf>=0.25)", plain.xyxy.tolist(), "red"),
            (f"640 tiles ({len(tiled)} boxes @conf>=0.25)", tiled.xyxy.tolist(), "cyan"),
        ]
        for col, (title, boxes, color) in enumerate(panels):
            ax = axes[row, col]
            ax.imshow(rgb)
            n_shown = 0
            for bx1, by1, bx2, by2 in boxes:
                if bx2 < x0 or bx1 > x1 or by2 < y0 or by1 > y1:
                    continue
                ax.add_patch(plt.Rectangle((bx1 - x0, by1 - y0), bx2 - bx1, by2 - by1, fill=False, color=color, lw=0.8))
                n_shown += 1
            ax.set_title(f"{title}; {n_shown} in this crop", fontsize=9)
            ax.axis("off")
    fig.suptitle(
        "Same weights, two ways of looking: resizing the frame to 640 px vs native-resolution tiles", fontsize=11
    )
    fig.tight_layout()
    fig.savefig(FIGS / "qualitative_plain_vs_tiled.png", dpi=130)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default=str(ROOT / "runs" / "y11n_640" / "weights" / "best.pt"))
    ap.add_argument("--data", type=Path, default=ROOT / "data" / "visdrone")
    ap.add_argument("--device", default="mps")
    ap.add_argument("--skip-qualitative", action="store_true")
    args = ap.parse_args()
    FIGS.mkdir(parents=True, exist_ok=True)
    evals = load_evals()
    if evals:
        fig_ap_by_size(evals)
        fig_per_class(evals)
    fig_training_curves()
    fig_explain_summary()
    if not args.skip_qualitative and Path(args.weights).exists():
        fig_qualitative(Path(args.weights), args.data, args.device)
    print("figures:", sorted(p.name for p in FIGS.glob("*.png")))


if __name__ == "__main__":
    main()
