#!/usr/bin/env python3
"""D-RISE explanations of true-positive detections on native-resolution val tiles, with
deletion/insertion faithfulness against a random-order baseline, stratified by object size.

Writes results/metrics/explain_drise.json and example panels in results/figures/explain/.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from smallobj.evaluate import match_predictions  # noqa: E402
from smallobj.explain import deletion_insertion, drise, saliency_stats  # noqa: E402
from smallobj.infer import load_model, predict_batch  # noqa: E402
from smallobj.tiling import crop, make_tiles  # noqa: E402
from smallobj.visdrone import CLASSES  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
BINS = {"tiny": (0, 16), "small": (16, 32), "medium": (32, 96), "large": (96, 10_000)}


def size_bin(w: float, h: float) -> str:
    side = float(np.sqrt(w * h))
    for name, (lo, hi) in BINS.items():
        if lo <= side < hi:
            return name
    return "large"


def select_candidates(
    model,
    coco_gt: dict,
    image_dir: Path,
    tile: int,
    overlap: float,
    n_per_bin: int,
    conf: float,
    device,
    seed: int,
    per_image: int = 2,
) -> list[dict]:
    """Walk shuffled val images; on each native-resolution tile match confident detections to
    GT boxes fully inside the tile; keep true positives until every size bin has ``n_per_bin``."""
    rng = np.random.default_rng(seed)
    anns = defaultdict(list)
    for a in coco_gt["annotations"]:
        anns[a["image_id"]].append(a)
    images = list(coco_gt["images"])
    rng.shuffle(images)
    quota = {b: n_per_bin for b in BINS}
    out: list[dict] = []
    for info in images:
        if all(q <= 0 for q in quota.values()):
            break
        image = cv2.imread(str(image_dir / info["file_name"]))
        gt = np.array(
            [
                [a["bbox"][0], a["bbox"][1], a["bbox"][0] + a["bbox"][2], a["bbox"][1] + a["bbox"][3]]
                for a in anns[info["id"]]
            ],
            dtype=float,
        ).reshape(-1, 4)
        gt_cls = np.array([a["category_id"] - 1 for a in anns[info["id"]]], dtype=int)
        taken_here = 0
        tiles = make_tiles(image.shape[1], image.shape[0], tile, overlap)
        rng.shuffle(tiles)
        for t in tiles:
            if taken_here >= per_image:
                break
            inside = (gt[:, 0] >= t.x0) & (gt[:, 1] >= t.y0) & (gt[:, 2] <= t.x1) & (gt[:, 3] <= t.y1)
            if not inside.any():
                continue
            g = gt[inside] - np.array([t.x0, t.y0, t.x0, t.y0])
            gc = gt_cls[inside]
            det = predict_batch(model, [crop(image, t)], tile, conf, 0.6, 300, device)[0]
            if len(det) == 0:
                continue
            match = match_predictions(g, gc, det.xyxy, det.cls, det.conf, 0.5)
            order = rng.permutation(len(det))
            for d in order:
                gi = match[d]
                if gi < 0:
                    continue
                b = size_bin(g[gi, 2] - g[gi, 0], g[gi, 3] - g[gi, 1])
                if quota[b] <= 0:
                    continue
                quota[b] -= 1
                taken_here += 1
                out.append(
                    {
                        "image": info["file_name"],
                        "tile": [t.x0, t.y0, t.x1, t.y1],
                        "det_xyxy": det.xyxy[d].round(1).tolist(),
                        "det_conf": float(det.conf[d]),
                        "cls": int(det.cls[d]),
                        "class_name": CLASSES[int(det.cls[d])],
                        "gt_xyxy": g[gi].round(1).tolist(),
                        "gt_side_px": float(np.sqrt((g[gi, 2] - g[gi, 0]) * (g[gi, 3] - g[gi, 1]))),
                        "size_bin": b,
                    }
                )
                if taken_here >= per_image:
                    break
    return out


def panel(crop_img: np.ndarray, sal: np.ndarray, rec: dict, curves: dict, path: Path) -> None:
    x1, y1, x2, y2 = (int(round(v)) for v in rec["det_xyxy"])
    pad = 80
    cx1, cy1 = max(x1 - pad, 0), max(y1 - pad, 0)
    cx2, cy2 = min(x2 + pad, crop_img.shape[1]), min(y2 + pad, crop_img.shape[0])
    rgb = cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB)
    fig, axes = plt.subplots(1, 4, figsize=(15, 3.9))
    axes[0].imshow(rgb)
    axes[0].add_patch(plt.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, color="lime", lw=1.5))
    axes[0].set_title(
        f"640 px tile: {rec['class_name']} conf {rec['det_conf']:.2f}\n"
        f"object side {rec['gt_side_px']:.0f} px ({rec['size_bin']})",
        fontsize=9,
    )
    axes[1].imshow(rgb)
    axes[1].imshow(sal, cmap="jet", alpha=0.45)
    axes[1].add_patch(plt.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, color="white", lw=1.2))
    axes[1].set_title("D-RISE saliency (whole tile)", fontsize=9)
    axes[2].imshow(rgb[cy1:cy2, cx1:cx2])
    axes[2].imshow(sal[cy1:cy2, cx1:cx2], cmap="jet", alpha=0.45)
    axes[2].add_patch(plt.Rectangle((x1 - cx1, y1 - cy1), x2 - x1, y2 - y1, fill=False, color="white", lw=1.2))
    axes[2].set_title("zoom around the target", fontsize=9)
    f = curves["saliency"]["fractions"]
    axes[3].plot(
        f,
        curves["saliency"]["deletion"],
        "r-",
        label=f"deletion, D-RISE order (AUC {curves['saliency']['deletion_auc']:.2f})",
    )
    axes[3].plot(
        f,
        curves["random"]["deletion"],
        "r:",
        label=f"deletion, random order (AUC {curves['random']['deletion_auc']:.2f})",
    )
    axes[3].plot(
        f,
        curves["saliency"]["insertion"],
        "b-",
        label=f"insertion, D-RISE order (AUC {curves['saliency']['insertion_auc']:.2f})",
    )
    axes[3].plot(
        f,
        curves["random"]["insertion"],
        "b:",
        label=f"insertion, random order (AUC {curves['random']['insertion_auc']:.2f})",
    )
    axes[3].set_xlabel("fraction of pixels removed / inserted")
    axes[3].set_ylabel("target score (normalised)")
    axes[3].set_ylim(-0.05, 1.3)
    axes[3].legend(fontsize=6.5, frameon=False)
    axes[3].set_title("faithfulness curves", fontsize=9)
    for ax in axes[:3]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default=str(ROOT / "runs" / "y11n_640" / "weights" / "best.pt"))
    ap.add_argument("--data", type=Path, default=ROOT / "data" / "visdrone")
    ap.add_argument("--device", default="mps")
    ap.add_argument("--tile", type=int, default=640)
    ap.add_argument("--n-per-bin", type=int, default=8)
    ap.add_argument("--n-masks", type=int, default=1000)
    ap.add_argument("--grid", type=int, default=16)
    ap.add_argument("--select-conf", type=float, default=0.25)
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--panels-per-bin", type=int, default=2)
    args = ap.parse_args()

    model = load_model(args.weights)
    coco_gt = json.loads((args.data / "annotations" / "val_coco.json").read_text())
    image_dir = args.data / "images" / "val"
    cands = select_candidates(
        model, coco_gt, image_dir, args.tile, 0.2, args.n_per_bin, args.select_conf, args.device, args.seed
    )
    print(
        f"selected {len(cands)} true-positive detections: "
        + ", ".join(f"{b}={sum(c['size_bin'] == b for c in cands)}" for b in BINS)
    )

    fig_dir = ROOT / "results" / "figures" / "explain"
    fig_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    records = []
    panels_done = defaultdict(int)
    for i, rec in enumerate(cands):
        image = cv2.imread(str(image_dir / rec["image"]))
        x0, y0, x1, y1 = rec["tile"]
        tile_img = image[y0:y1, x0:x1]
        target = np.array(rec["det_xyxy"])
        res = drise(
            model, tile_img, target, rec["cls"], args.tile, args.n_masks, args.grid, 0.5, 16, 0.01, args.device, rng
        )
        curves = {
            how: deletion_insertion(
                model,
                tile_img,
                res.saliency,
                target,
                rec["cls"],
                args.tile,
                how,
                args.steps,
                16,
                0.01,
                args.device,
                rng,
            )
            for how in ("saliency", "random")
        }
        st = saliency_stats(res.saliency, target)
        rec_out = {**rec, "base_score": res.base_score, **st, "curves": curves}
        records.append(rec_out)
        if panels_done[rec["size_bin"]] < args.panels_per_bin:
            panels_done[rec["size_bin"]] += 1
            panel(
                tile_img,
                res.saliency,
                rec,
                curves,
                fig_dir / f"{rec['size_bin']}_{panels_done[rec['size_bin']]}_{rec['class_name']}.png",
            )
        print(
            f"[{i + 1}/{len(cands)}] {rec['size_bin']:6s} {rec['class_name']:15s} side={rec['gt_side_px']:5.1f}px "
            f"del AUC drise={curves['saliency']['deletion_auc']:.2f} rand={curves['random']['deletion_auc']:.2f} | "
            f"ins AUC drise={curves['saliency']['insertion_auc']:.2f} rand={curves['random']['insertion_auc']:.2f} | "
            f"mass_in_box={st['mass_in_box']:.2f} area90={st['area_frac_90']:.3f}"
        )

    summary = {}
    for b in BINS:
        rs = [r for r in records if r["size_bin"] == b]
        if not rs:
            continue

        def m(key_fn, rs=rs):
            vals = np.array([key_fn(r) for r in rs], dtype=float)
            return {"mean": float(vals.mean()), "std": float(vals.std(ddof=0)), "n": int(len(vals))}

        summary[b] = {
            "n": len(rs),
            "mean_side_px": float(np.mean([r["gt_side_px"] for r in rs])),
            "deletion_auc_drise": m(lambda r: r["curves"]["saliency"]["deletion_auc"]),
            "deletion_auc_random": m(lambda r: r["curves"]["random"]["deletion_auc"]),
            "insertion_auc_drise": m(lambda r: r["curves"]["saliency"]["insertion_auc"]),
            "insertion_auc_random": m(lambda r: r["curves"]["random"]["insertion_auc"]),
            "mass_in_box": m(lambda r: r["mass_in_box"]),
            "box_area_frac": m(lambda r: r["box_area_frac"]),
            "area_frac_90": m(lambda r: r["area_frac_90"]),
        }
    out = {
        "settings": {
            "weights": args.weights,
            "tile": args.tile,
            "n_masks": args.n_masks,
            "grid": args.grid,
            "p": 0.5,
            "steps": args.steps,
            "select_conf": args.select_conf,
            "n_per_bin": args.n_per_bin,
            "seed": args.seed,
            "fill": "gaussian blur (51 px) of the tile",
        },
        "summary": summary,
        "records": records,
    }
    (ROOT / "results" / "metrics" / "explain_drise.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
