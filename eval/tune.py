"""Threshold tuning on the VALIDATION split only, then frozen to results/thresholds.json.  Owner: Kriti.
  python eval/tune.py            (refuses to overwrite an existing thresholds.json; --force to redo)

Each threshold grid = 21 quantiles of that score on the validation split (config thresholds.grid_points).
Thresholds are tuned per dataset (KnowHalBench pooled over its 3 settings, RAGTruth over its 3 tasks) to
maximise F1. The controlled set reuses the HaluEval thresholds (config thresholds.controlled_set_uses).
Every method (ours, ablations, baselines) follows the same Method interface defined here.
"""
import argparse
import datetime
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common import CACHE, CFG, RESULTS, load_claims  # noqa: E402

TUNED_DATASETS = ["halueval", "khb", "ragtruth"]


def grid(x, k):
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    return np.unique(np.quantile(x, np.linspace(0, 1, k)))


def _f1(pred, y):
    """pred (m, n) bool, y (n,) bool -> F1 for each of the m rows."""
    tp, fp, fn = (pred & y).sum(1), (pred & ~y).sum(1), (~pred & y).sum(1)
    den = 2 * tp + fp + fn
    return np.where(den > 0, 2 * tp / np.maximum(den, 1), 0.0)


def best_single(score, y, k):
    """flag = score > tau"""
    ts = grid(score, k)
    f = _f1(score[None, :] > ts[:, None], y)
    i = int(np.argmax(f))
    return dict(tau=float(ts[i]), val_f1=float(f[i]))


def best_pair(g, sA, y, k):
    """CRAG-style: flag = (g >= tau_g) AND (s_A > tau_A)"""
    TA, best = grid(sA, k), None
    for tg in grid(g, k):
        f = _f1((g >= tg)[None, :] & (sA[None, :] > TA[:, None]), y)
        i = int(np.argmax(f))
        if best is None or f[i] > best["val_f1"]:
            best = dict(tau_g=float(tg), tau_A=float(TA[i]), val_f1=float(f[i]))
    return best


def best_triple(g, sA, sB, y, k, use_disagreement=True, fixed_tau_g=None):
    """Our diagnosis. Any flag = A | B | D = (trusted & s_A > tau_A) | (s_B > tau_B);
    without the disagreement rule it is (trusted & s_A > tau_A) | (untrusted & s_B > tau_B)."""
    TB = grid(sB, k)
    Bhit = sB[None, :] > TB[:, None]
    best = None
    for tg in ([fixed_tau_g] if fixed_tau_g is not None else grid(g, k)):
        trusted = g >= tg
        for ta in grid(sA, k):
            A = (trusted & (sA > ta))[None, :]
            pred = (A | Bhit) if use_disagreement else (A | (~trusted[None, :] & Bhit))
            f = _f1(pred, y)
            i = int(np.argmax(f))
            if best is None or f[i] > best["val_f1"]:
                best = dict(tau_g=float(tg), tau_A=float(ta), tau_B=float(TB[i]), val_f1=float(f[i]))
    return best


class Method:
    """name, needs (score columns), prep(df), risk(df) for AUROC (higher = more likely hallucinated),
    tune(df_val, k) -> params dict, predict(df, params) -> bool array."""
    name = ""
    needs = []

    def prep(self, df):
        return df

    def usable(self, df):
        return df[self.needs].notna().all(axis=1).values

    def risk(self, df):
        raise NotImplementedError

    def tune(self, v, k):
        raise NotImplementedError

    def predict(self, df, params):
        raise NotImplementedError


def main_methods():
    from baselines.crag_style import METHOD as crag
    from baselines.refchecker_style import METHOD as refchecker
    from baselines.saplma_style import METHOD as saplma
    from baselines.selfcheck import METHOD as selfcheck
    from pipeline.diagnose import Ours
    return [Ours(), selfcheck, saplma, refchecker, crag]


def ablation_methods():
    from pipeline.diagnose import Ours
    return [Ours(), Ours("gate_always_A", g_mode=1), Ours("gate_always_B", g_mode=0),
            Ours("no_disagreement_rule", use_disagreement=False), Ours("no_clipping", pcol="p_know_noclip")]


def load_frame():
    claims = load_claims()[["claim_id", "dataset", "setting", "example_id", "split", "label"]]
    return claims.merge(pd.read_parquet(CACHE / "scores.parquet"), on="claim_id", how="left")


def main(force=False):
    path = RESULTS / "thresholds.json"
    if path.exists() and not force:
        sys.exit("thresholds.json already exists (frozen). Use --force only if you will redo every test table.")
    df, k = load_frame(), CFG["thresholds"]["grid_points"]
    methods = {m.name: m for m in main_methods() + ablation_methods()}.values()
    out = {}
    for m in methods:
        if any(c not in df.columns for c in m.needs):
            print("skip %-22s (missing columns %s)" % (m.name, m.needs))
            continue
        out[m.name] = {}
        for ds in TUNED_DATASETS:
            v = m.prep(df[(df.dataset == ds) & (df.split == "validation")])
            v = v[m.usable(v)].reset_index(drop=True)
            if len(v) == 0 or v.label.nunique() < 2:
                print("skip %-22s %-9s (no usable validation rows)" % (m.name, ds))
                continue
            out[m.name][ds] = dict(m.tune(v, k), n_val=len(v))
            print("%-22s %-9s %s" % (m.name, ds, {a: round(b, 3) if isinstance(b, float) else b for a, b in out[m.name][ds].items()}))
    out["_meta"] = dict(date=datetime.date.today().isoformat(), grid_points=k, tuned_on="validation split only")
    json.dump(out, open(path, "w"), indent=2)
    print("\nFROZEN ->", path, "(commit it; test-split tables may be built now)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    main(ap.parse_args().force)
