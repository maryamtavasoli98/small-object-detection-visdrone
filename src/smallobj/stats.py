"""Dataset statistics and figures from the per-box CSV written during conversion."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from .visdrone import CLASSES  # noqa: E402

BIN_EDGES = [0.0, 16.0, 32.0, 96.0, np.inf]  # object side in px = sqrt(area)
BIN_NAMES = ["tiny (<16 px)", "small (16-32)", "medium (32-96)", "large (>96)"]


def load_boxes(meta_dir: Path, splits: tuple[str, ...]) -> pd.DataFrame:
    frames = []
    for s in splits:
        df = pd.read_csv(meta_dir / f"{s}_boxes.csv")
        df["split"] = s
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)
    df["side"] = np.sqrt(df["area"].astype(float))
    df["size_bin"] = pd.cut(df["side"], bins=BIN_EDGES, labels=BIN_NAMES, right=False)
    return df


def summarise(df: pd.DataFrame) -> dict:
    out: dict = {}
    for split, g in df.groupby("split"):
        per_img = g.groupby("image").size()
        out[split] = {
            "images": int(per_img.size),
            "boxes": int(len(g)),
            "objects_per_image": {
                "mean": float(per_img.mean()),
                "median": float(per_img.median()),
                "max": int(per_img.max()),
            },
            "share_by_size_bin": {
                k: float(v) for k, v in (g["size_bin"].value_counts(normalize=True).sort_index()).items()
            },
            "median_side_px_by_class": {c: float(v) for c, v in g.groupby("name")["side"].median().items()},
            "class_counts": {c: int(v) for c, v in g["name"].value_counts().items()},
            "occluded_share_by_size_bin": {
                k: float(v)
                for k, v in g.groupby("size_bin", observed=True)["occlusion"].apply(lambda s: (s > 0).mean()).items()
            },
            "image_size_modes": [
                f"{w}x{h}"
                for (w, h), _ in g.groupby(["img_w", "img_h"]).size().sort_values(ascending=False).head(3).items()
            ],
        }
    return out


def fig_size_hist(df: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 3.6))
    bins = np.logspace(0, np.log10(600), 60)
    for split, g in df.groupby("split"):
        ax.hist(g["side"], bins=bins, alpha=0.6, label=f"{split} ({len(g):,} boxes)", density=True)
    for x, lbl in [(16, "16 px"), (32, "32 px"), (96, "96 px")]:
        ax.axvline(x, color="k", ls="--", lw=0.8)
        ax.text(x, ax.get_ylim()[1] * 0.92, f" {lbl}", fontsize=8)
    ax.set_xscale("log")
    ax.set_xlabel("object side  sqrt(w*h)  [px, original resolution]")
    ax.set_ylabel("density")
    ax.set_title("VisDrone-DET: most objects are below the COCO 'small' threshold (32 px)")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def fig_size_by_class(df: pd.DataFrame, path: Path, split: str = "train") -> None:
    g = df[df["split"] == split]
    table = pd.crosstab(g["name"], g["size_bin"], normalize="index").reindex(CLASSES)
    counts = g["name"].value_counts().reindex(CLASSES)
    fig, ax = plt.subplots(figsize=(8, 3.8))
    bottom = np.zeros(len(CLASSES))
    colors = ["#b2182b", "#ef8a62", "#67a9cf", "#2166ac"]
    for col, c in zip(table.columns, colors, strict=True):
        ax.bar(range(len(CLASSES)), table[col].values, bottom=bottom, label=col, color=c)
        bottom += table[col].values
    ax.set_xticks(range(len(CLASSES)))
    ax.set_xticklabels([f"{c}\n(n={counts[c]:,})" for c in CLASSES], fontsize=7)
    ax.set_ylabel("share of boxes")
    ax.set_title(f"Object-size mix per class ({split})")
    ax.legend(ncol=4, fontsize=8, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.25))
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def fig_objects_per_image(df: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 3.2))
    for split, g in df.groupby("split"):
        ax.hist(g.groupby("image").size(), bins=50, alpha=0.6, label=split, density=True)
    ax.set_xlabel("annotated objects per image")
    ax.set_ylabel("density")
    ax.set_title("Crowded scenes: objects per image")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def run(meta_dir: Path, splits: tuple[str, ...], metrics_out: Path, figures_dir: Path) -> dict:
    df = load_boxes(meta_dir, splits)
    summary = summarise(df)
    figures_dir.mkdir(parents=True, exist_ok=True)
    metrics_out.parent.mkdir(parents=True, exist_ok=True)
    fig_size_hist(df, figures_dir / "dataset_object_sizes.png")
    fig_size_by_class(df, figures_dir / "dataset_size_by_class.png", split=splits[0])
    fig_objects_per_image(df, figures_dir / "dataset_objects_per_image.png")
    metrics_out.write_text(json.dumps(summary, indent=2))
    return summary
