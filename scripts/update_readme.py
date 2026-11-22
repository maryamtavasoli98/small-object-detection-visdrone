#!/usr/bin/env python3
"""Regenerate the result tables inside README.md from results/metrics/*.json.

README.md contains blocks delimited by ``<!-- table:NAME -->`` ... ``<!-- /table:NAME -->``;
everything between the markers is replaced. Numbers in the README therefore always come from
files written by the experiment scripts.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from smallobj.visdrone import CLASSES

ROOT = Path(__file__).resolve().parents[1]
METRICS = ROOT / "results" / "metrics"
README = ROOT / "README.md"

PRETTY = {
    "y11n_640_plain": ("YOLO11n @640", "resize whole frame to 640"),
    "y11n_640_plain1024": ("YOLO11n @640", "resize whole frame to 1024 (test-time only)"),
    "y11n_640_tiled": ("YOLO11n @640", "640 px tiles, 20 % overlap"),
    "y11n_640_tiled_full": ("YOLO11n @640", "640 px tiles + full frame"),
    "y11n_1024_plain": ("YOLO11n @1024", "resize whole frame to 1024"),
    "y11n_1024_tiled": ("YOLO11n @1024", "1024 px tiles, 20 % overlap"),
    "y11n_1024_tiled640": ("YOLO11n @1024", "640 px tiles + full frame"),
}


def f3(x) -> str:
    return "–" if x is None or x != x else f"{x:.3f}"


def pct(x) -> str:
    return "–" if x is None or x != x else f"{100 * x:.1f} %"


def table_dataset() -> str:
    p = METRICS / "dataset_stats.json"
    if not p.exists():
        return "_run `make stats`_"
    s = json.loads(p.read_text())
    rows = [
        "| split | images | boxes | objects / image (median, max) | tiny <16 px | small 16-32 | medium 32-96 | large >96 |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for split in ("train", "val"):
        if split not in s:
            continue
        d = s[split]
        sb = d["share_by_size_bin"]
        keys = list(sb)
        rows.append(
            f"| {split} | {d['images']:,} | {d['boxes']:,} | {d['objects_per_image']['median']:.0f}, {d['objects_per_image']['max']} | "
            + " | ".join(pct(sb[k]) for k in keys)
            + " |"
        )
    return "\n".join(rows)


def table_training() -> str:
    runs = sorted(METRICS.glob("train_*.json"))
    if not runs:
        return "_run `make train`_"
    rows = [
        "| run | init | input | epochs | batch | time | val mAP50 | val mAP50-95 |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for p in runs:
        r = json.loads(p.read_text())
        hours = r["train_seconds"] / 3600
        rows.append(
            f"| `{r['name']}` | {Path(r['init']).name} | {r['imgsz']} px | {r['epochs_run']} | {r['batch']} | {hours:.1f} h | "
            f"{f3(r['val_mAP50_ultralytics'])} | {f3(r['val_mAP50_95_ultralytics'])} |"
        )
    rows.append("")
    rows.append(
        "_mAP from the Ultralytics validator (its own protocol, no ignored-region filtering); the tables below use the COCO protocol described in Evaluation._"
    )
    return "\n".join(rows)


def table_main() -> str:
    evals = {p.stem.removeprefix("eval_"): json.loads(p.read_text()) for p in sorted(METRICS.glob("eval_*.json"))}
    if not evals:
        return "_run `make eval`_"
    rows = [
        "| weights | inference | AP | AP50 | AP50 tiny | AP50 small | AP50 medium | AP50 large | AR | s / image |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for name in PRETTY:
        if name not in evals:
            continue
        e = evals[name]
        o, s, c = e["overall"], e["by_size"], e["config"]
        w, mode = PRETTY[name]
        rows.append(
            f"| {w} | {mode} | {f3(o['AP'])} | **{f3(o['AP50'])}** | {f3(s['tiny']['AP50'])} | {f3(s['small']['AP50'])} | "
            f"{f3(s['medium']['AP50'])} | {f3(s['large']['AP50'])} | {f3(o['AR'])} | {c['seconds_per_image']:.3f} |"
        )
    any_eval = next(iter(evals.values()))
    n = any_eval["by_size"]
    rows.append("")
    rows.append(
        f"_VisDrone val, {any_eval['config']['images']} images. Ground-truth boxes per bin: tiny {n['tiny']['n_gt']:,}, small {n['small']['n_gt']:,}, "
        f"medium {n['medium']['n_gt']:,}, large {n['large']['n_gt']:,}. Timing on an Apple M3 Pro (MPS), batch of tiles = 8._"
    )
    return "\n".join(rows)


def table_per_class(a: str = "y11n_640_plain", b: str = "y11n_640_tiled") -> str:
    pa, pb = METRICS / f"eval_{a}.json", METRICS / f"eval_{b}.json"
    if not (pa.exists() and pb.exists()):
        return "_run `make eval`_"
    ea, eb = json.loads(pa.read_text()), json.loads(pb.read_text())
    stats = (
        json.loads((METRICS / "dataset_stats.json").read_text())["val"]["median_side_px_by_class"]
        if (METRICS / "dataset_stats.json").exists()
        else {}
    )
    rows = ["| class | median side (px) | AP50 plain 640 | AP50 tiled 640 | change |", "|---|---|---|---|---|"]
    for c in CLASSES:
        x, y = ea["per_class"][c]["AP50"], eb["per_class"][c]["AP50"]
        rows.append(f"| {c} | {stats.get(c, float('nan')):.0f} | {f3(x)} | {f3(y)} | {y - x:+.3f} |")
    return "\n".join(rows)


def table_explain() -> str:
    p = METRICS / "explain_drise.json"
    if not p.exists():
        return "_run `make explain`_"
    d = json.loads(p.read_text())
    s = d["summary"]
    rows = [
        "| object size | n | mean side (px) | deletion AUC D-RISE ↓ | deletion AUC random | insertion AUC D-RISE ↑ | insertion AUC random | saliency mass inside box | box share of tile | tile share holding 90 % of mass |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for b in ("tiny", "small", "medium", "large"):
        if b not in s:
            continue
        r = s[b]
        rows.append(
            f"| {b} | {r['n']} | {r['mean_side_px']:.0f} | {r['deletion_auc_drise']['mean']:.2f} ± {r['deletion_auc_drise']['std']:.2f} | "
            f"{r['deletion_auc_random']['mean']:.2f} ± {r['deletion_auc_random']['std']:.2f} | {r['insertion_auc_drise']['mean']:.2f} ± {r['insertion_auc_drise']['std']:.2f} | "
            f"{r['insertion_auc_random']['mean']:.2f} ± {r['insertion_auc_random']['std']:.2f} | {pct(r['mass_in_box']['mean'])} | {pct(r['box_area_frac']['mean'])} | {pct(r['area_frac_90']['mean'])} |"
        )
    st = d["settings"]
    rows.append("")
    rows.append(
        f"_True-positive detections (conf ≥ {st['select_conf']}) on native-resolution {st['tile']} px val tiles, {st['n_masks']} masks on a {st['grid']}×{st['grid']} grid, "
        f"{st['steps']} deletion/insertion steps, removed pixels replaced by a blurred copy. Mean ± std over n detections._"
    )
    return "\n".join(rows)


TABLES = {
    "dataset": table_dataset,
    "training": table_training,
    "main": table_main,
    "per_class": table_per_class,
    "explain": table_explain,
}


def main() -> None:
    text = README.read_text()
    for name, fn in TABLES.items():
        pattern = re.compile(rf"(<!-- table:{name} -->\n).*?(\n<!-- /table:{name} -->)", re.S)
        if not pattern.search(text):
            print(f"marker for {name} not found")
            continue
        text = pattern.sub(lambda m, fn=fn: m.group(1) + fn() + m.group(2), text)
    README.write_text(text)
    print("README tables updated")


if __name__ == "__main__":
    main()
