"""Merge all claim files and assign validation/test splits BY EXAMPLE (never by claim).

Reads cache/claims_{halueval,controlled,ragtruth,khb}.parquet that exist, assigns
`split` = validation (30%) / test (70%) with seed 42, and writes cache/claims.parquet.
Splits are made per (dataset, setting) so every KnowHalBench context setting and every
RAGTruth task has both validation and test examples. The controlled set copies the split
of the HaluEval question it came from (same example_id), so there is no leakage.
Run from the project root:  python data/splits.py   (after the four loaders)
"""
import random
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parent.parent
CFG = yaml.safe_load(open(ROOT / "config.yaml", encoding="utf-8"))
CACHE = ROOT / CFG["project"]["paths"]["cache"]
SEED = CFG["project"]["seed"]
VAL = CFG["project"]["val_fraction"]
FILES = ["halueval", "controlled", "ragtruth", "khb"]


def assign_splits(df):
    mapping = {}
    base = df[df.dataset != "controlled"]
    for _, grp in base.groupby(["dataset", "setting"]):
        ids = sorted(grp.example_id.unique())
        random.Random(SEED).shuffle(ids)
        n_val = int(round(VAL * len(ids)))
        for i, ex in enumerate(ids):
            mapping[ex] = "validation" if i < n_val else "test"
    ctrl = df[df.dataset == "controlled"]
    missing = set(ctrl.example_id) - set(mapping)
    if missing:
        raise SystemExit("controlled set has %d questions that are not in the HaluEval claims" % len(missing))
    out = df.copy()
    out["split"] = out.example_id.map(mapping)
    assert out.split.notna().all()
    return out


if __name__ == "__main__":
    frames = []
    for name in FILES:
        path = CACHE / ("claims_%s.parquet" % name)
        if path.exists():
            frames.append(pd.read_parquet(path))
            print("loaded", path.name, len(frames[-1]))
        else:
            print("not found (skipped):", path.name)
    df = assign_splits(pd.concat(frames, ignore_index=True))
    assert df.claim_id.is_unique, "duplicate claim_id"
    df.to_parquet(CACHE / "claims.parquet", index=False)
    print("\nsaved %d claims -> %s" % (len(df), CACHE / "claims.parquet"))
    print(df.groupby(["dataset", "setting", "split"]).agg(claims=("label", "size"),
          positive_share=("label", "mean")).round(3).to_string())
