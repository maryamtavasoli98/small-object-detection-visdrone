#!/usr/bin/env python3
"""Dataset statistics (object sizes, class mix, crowding) and the dataset figures."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from smallobj.stats import run

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=ROOT / "data" / "visdrone")
    ap.add_argument("--splits", nargs="+", default=["train", "val"])
    args = ap.parse_args()
    summary = run(
        args.data / "meta",
        tuple(args.splits),
        ROOT / "results" / "metrics" / "dataset_stats.json",
        ROOT / "results" / "figures",
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
