"""Result tables on the TEST split, using the frozen thresholds.  Owner: Kriti.
  python eval/tables.py          (needs results/thresholds.json from eval/tune.py; never tunes anything)
Writes results/main_table.csv, per_setting_table.csv, controlled_table.csv, ablation_table.csv, latency_table.csv.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common import CACHE, CFG, RESULTS  # noqa: E402
from eval.metrics import auroc, bootstrap_ci, confusion_metrics  # noqa: E402
from eval.tune import TUNED_DATASETS, ablation_methods, load_frame, main_methods  # noqa: E402
from pipeline.diagnose import Ours, diagnose  # noqa: E402

COLS = ["n", "positives", "accuracy", "precision", "recall", "f1", "auroc"]


def evaluate(m, d, params, ci=False):
    d = m.prep(d)
    d = d[m.usable(d)].reset_index(drop=True)
    if len(d) == 0:
        return None
    y, score, pred = d.label.values.astype(int), m.risk(d), m.predict(d, params)
    res = dict(n=len(d), positives=int(y.sum()), auroc=auroc(y, score))
    res.update({k: v for k, v in confusion_metrics(y, pred).items() if k in ("accuracy", "precision", "recall", "f1")})
    if ci:
        g = d.example_id.values
        res["f1_lo"], res["f1_hi"] = bootstrap_ci(g, lambda i: confusion_metrics(y[i], pred[i])["f1"])
        res["auroc_lo"], res["auroc_hi"] = bootstrap_ci(g, lambda i: auroc(y[i], score[i]))
    return res


def show(title, df):
    print("\n== %s ==" % title)
    print(df.round(3).to_string(index=False))


def main():
    path = RESULTS / "thresholds.json"
    if not path.exists():
        sys.exit("results/thresholds.json not found. Run python eval/tune.py first (validation split only).")
    thr = json.load(open(path))
    test = load_frame()
    test = test[test.split == "test"]
    ids = RESULTS / "halueval_overlap_ids.txt"
    overlap = set(ids.read_text().split()) if ids.exists() else set()

    # 1. main comparison, per dataset, with bootstrap CIs (+ HaluEval without the answer-overlap questions)
    rows = []
    for name in TUNED_DATASETS + ["halueval_no_overlap"]:
        base = "halueval" if name == "halueval_no_overlap" else name
        d = test[test.dataset == base]
        if name == "halueval_no_overlap":
            d = d[~d.example_id.isin(overlap)]
        for m in main_methods():
            p = thr.get(m.name, {}).get(base)
            r = evaluate(m, d, p, ci=True) if p else None
            if r:
                rows.append(dict(dataset=name, method=m.name, **r))
    main_t = pd.DataFrame(rows)
    main_t.to_csv(RESULTS / "main_table.csv", index=False)
    show("main comparison (test split)", main_t)

    # 2. per context setting / task, with the dataset-level thresholds
    rows = []
    for base in TUNED_DATASETS:
        for st, d in test[test.dataset == base].groupby("setting"):
            for m in main_methods():
                p = thr.get(m.name, {}).get(base)
                r = evaluate(m, d, p) if p else None
                if r:
                    rows.append(dict(dataset=base, setting=st, method=m.name, **r))
    per = pd.DataFrame(rows)
    per.to_csv(RESULTS / "per_setting_table.csv", index=False)
    show("per setting / task (test split)", per)

    # 3. controlled set: disagreement flag, positive class = unreliable reference
    m, ds = Ours(), CFG["thresholds"]["controlled_set_uses"]
    p = thr["ours"][ds]
    d = m.prep(test[test.dataset == "controlled"])
    d = d[m.usable(d)].reset_index(drop=True)
    pred, flag_d, kind = diagnose(d, p["tau_g"], p["tau_A"], p["tau_B"])
    f = RESULTS / "controlled_aligned_ids.txt"
    aligned = set(f.read_text().split()) if f.exists() else set()
    rows = []
    for name, mask in [("all questions", np.ones(len(d), bool)), ("aligned questions only", d.example_id.isin(aligned).values)]:
        if mask.sum():
            rows.append(dict(subset=name, n=int(mask.sum()), positives=int(d.label.values[mask].sum()),
                             auroc=auroc(d.label.values[mask], d["d"].values[mask]),
                             **{k: v for k, v in confusion_metrics(d.label.values[mask], flag_d[mask]).items()
                                if k in ("accuracy", "precision", "recall", "f1")}))
    ctl = pd.DataFrame(rows)
    ctl.to_csv(RESULTS / "controlled_table.csv", index=False)
    show("controlled set: disagreement flag, positive = unreliable reference (thresholds from %s)" % ds, ctl)
    rate = pd.DataFrame(dict(setting=d.setting, disagreement=flag_d, flag_A=kind == "A", flag_B=kind == "B")).groupby("setting").mean()
    print("\nshare of claims flagged per setting (unreliable should be high in 'disagreement', the other two low):")
    print(rate.round(3).to_string())

    # 4. ablations (each variant has its own thresholds tuned on validation)
    rows = []
    for base in TUNED_DATASETS:
        for m in ablation_methods():
            p = thr.get(m.name, {}).get(base)
            r = evaluate(m, test[test.dataset == base], p) if p else None
            if r:
                rows.append(dict(dataset=base, variant=m.name, **r))
    abl = pd.DataFrame(rows)
    abl.to_csv(RESULTS / "ablation_table.csv", index=False)
    show("ablations (test split)", abl)

    # 5. latency (from score_all.py runs). Use the reference laptop's files for the report.
    lat = [json.loads(x.read_text()) for x in sorted(CACHE.glob("latency_*.json"))]
    if lat:
        lt = pd.DataFrame(lat)
        lt.to_csv(RESULTS / "latency_table.csv", index=False)
        show("latency (state which laptop is the reference)", lt)
    print("\nlayer sweep, per-topic probe table and the grid: results/probe_*.csv (Michelle)")


if __name__ == "__main__":
    main()
