"""Sanity check of the merged scores BEFORE tuning (validation split only; never touches the test split).
  python scripts/sanity_scores.py
Prints the missing values per score column, then the validation AUROC of each raw risk score per dataset and
setting: Path A (1 - a_entail), Path B (1 - p_know), Path B without clipping (1 - p_know_noclip) and SelfCheckGPT.
Healthy values are above 0.5 (KnowHalBench Zero for Path A should be near the 0.90 of the triplet test).
A value clearly below 0.5 means a sign or merge error: fix it before eval/tune.py freezes the thresholds.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from eval.metrics import auroc  # noqa: E402
from eval.tune import load_frame  # noqa: E402

# name -> (score column, True if the risk is 1 - column)
SCORES = {"path_A": ("a_entail", True), "path_B": ("p_know", True), "path_B_noclip": ("p_know_noclip", True),
          "selfcheck": ("selfcheck", False)}

d = load_frame()
cols = [c for c in ["a_entail", "a_contra", "g", "p_know", "p_know_noclip", "selfcheck"] if c in d.columns]
print("rows: %d" % len(d))
print("missing values per column (selfcheck should be missing only for the %d controlled claims):" % (d.dataset == "controlled").sum())
print(d[cols].isna().sum().to_string())
for name, (col, _) in SCORES.items():
    if col not in d.columns:
        print("NOTE: column %s is not in scores.parquet (merge it first)" % col)

v = d[(d.split == "validation") & (d.dataset != "controlled")]
rows = []
for (ds, st), g in v.groupby(["dataset", "setting"]):
    row = {"dataset": ds, "setting": st, "n": len(g), "positives": int(g.label.sum())}
    for name, (col, flip) in SCORES.items():
        if col not in g.columns:
            row[name] = np.nan
            continue
        h = g[g[col].notna()]
        row[name] = auroc(h.label.values, (1 - h[col] if flip else h[col]).values) if len(h) else np.nan
    row["mean_g"] = g["g"].mean() if "g" in g.columns else np.nan
    rows.append(row)
res = pd.DataFrame(rows)
print("\nvalidation AUROC of each raw risk score (controlled set excluded):")
print(res.round(3).to_string(index=False))
for _, r in res.iterrows():
    for name in SCORES:
        if r[name] == r[name] and r[name] < 0.5:
            print("CHECK: %s %s %s AUROC %.3f is below 0.5 (small samples can do this; a large gap means a sign or merge error)"
                  % (name, r["dataset"], r["setting"], r[name]))
