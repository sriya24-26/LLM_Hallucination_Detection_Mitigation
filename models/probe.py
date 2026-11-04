"""Path B: hidden-state probe on a small open-weight model.  Owner: Michelle.
  python models/probe.py extract      hidden states (final token, all layers) of truefalse + claims -> cache/
  python models/probe.py sweep        leave-one-topic-out: layer sweep, then clipping x classifier grid at the best layer
  python models/probe.py final        train the final probes on all statements, score claims -> cache/scores_pathb.parquet
  python models/probe.py all          the three steps in order
Options: --model Qwen/Qwen2.5-1.5B-Instruct   (comparison run; use with --only truefalse, then sweep)

Probe = per-dimension percentile clipping (INSIDE-style) -> StandardScaler -> MLP or logistic regression.
It predicts p_know = P(statement is true). Path B risk is s_B = 1 - p_know.
p_know_noclip (same layer and classifier, no clipping) is the SAPLMA-style baseline and the no-clipping ablation.
"""
import argparse
import json
import sys
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common import CACHE, CFG, RESULTS, SEED, load_claims, load_truefalse  # noqa: E402

warnings.filterwarnings("ignore", category=ConvergenceWarning)
P = CFG["probe"]


def model_tag(name):
    return name.split("/")[-1].lower()


class Extractor:
    """Final-token hidden state of every layer (embeddings + all transformer layers), left padding."""

    def __init__(self, name):
        import torch
        from transformers import AutoModel, AutoTokenizer

        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(name, padding_side="left", truncation_side="left")
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        self.model = AutoModel.from_pretrained(name, torch_dtype=torch.float32).eval()

    def __call__(self, texts, batch_size=16, max_length=256):
        n = len(texts)
        order = np.argsort([len(t) for t in texts])  # similar lengths together = less padding
        out = None
        with self.torch.inference_mode():
            for b in tqdm(range(0, n, batch_size), desc="extract"):
                idx = order[b:b + batch_size]
                enc = self.tok([texts[i] for i in idx], return_tensors="pt", padding=True,
                               truncation=True, max_length=max_length)
                hs = self.model(**enc, output_hidden_states=True).hidden_states
                feats = self.torch.stack([h[:, -1, :] for h in hs], dim=1).numpy()  # (batch, layers+1, dim)
                if out is None:
                    out = np.zeros((n,) + feats.shape[1:], np.float32)
                out[idx] = feats
        return out


def make_clf(kind, C=1.0):
    g = P["classifier_grid"]
    if kind == "logreg":
        return LogisticRegression(C=C, max_iter=1000)
    return MLPClassifier(hidden_layer_sizes=tuple(g["mlp_hidden"]), learning_rate_init=g["mlp_lr"], alpha=g["mlp_l2"],
                         early_stopping=True, max_iter=200, random_state=SEED)


class Probe:
    def __init__(self, clip=None, kind="mlp", C=1.0):
        self.clip, self.kind, self.C = clip, kind, C

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float32)
        if self.clip is not None:  # clip bounds come from the training rows only
            self.lo, self.hi = np.percentile(X, [self.clip, 100 - self.clip], axis=0)
            X = np.clip(X, self.lo, self.hi)
        self.scaler = StandardScaler().fit(X)
        self.clf = make_clf(self.kind, self.C).fit(self.scaler.transform(X), y)
        return self

    def p_true(self, X):
        X = np.asarray(X, dtype=np.float32)
        if self.clip is not None:
            X = np.clip(X, self.lo, self.hi)
        return self.clf.predict_proba(self.scaler.transform(X))[:, 1]


def loto(X, y, topics, **kw):
    """Leave-one-topic-out AUROC per topic (y = 1 for true statements)."""
    out = {}
    for t in sorted(set(topics)):
        te = topics == t
        out[t] = roc_auc_score(y[te], Probe(**kw).fit(X[~te], y[~te]).p_true(X[te]))
    return out


def cmd_extract(a):
    tag, ex = model_tag(a.model), Extractor(a.model)
    if a.only in ("truefalse", "both"):
        tf = load_truefalse()
        np.save(CACHE / f"hidden_{tag}_truefalse.npy", ex(tf.statement.tolist(), a.batch_size))
        print("saved hidden states for %d statements" % len(tf))
    if a.only in ("claims", "both"):
        cl = load_claims()
        np.save(CACHE / f"hidden_{tag}_claims.npy", ex(cl.claim_text.tolist(), a.batch_size))
        np.save(CACHE / f"hidden_{tag}_claims_ids.npy", np.array(cl.claim_id.tolist(), dtype=str))
        print("saved hidden states for %d claims" % len(cl))


def cmd_sweep(a):
    tag = model_tag(a.model)
    H = np.load(CACHE / f"hidden_{tag}_truefalse.npy", mmap_mode="r")
    tf = load_truefalse()
    y, topics = tf.label_true.values, tf.topic.values

    rows = []
    for L in tqdm(range(H.shape[1]), desc="layer sweep (logreg, no clipping)"):
        r = loto(np.asarray(H[:, L, :]), y, topics, clip=None, kind="logreg", C=1.0)
        rows.append(dict(layer=L, mean_auroc=float(np.mean(list(r.values()))), **r))
    sweep = pd.DataFrame(rows)
    sweep.to_csv(RESULTS / f"probe_layer_sweep_{tag}.csv", index=False)
    print(sweep.round(3).to_string(index=False))
    best = int(sweep.loc[sweep.mean_auroc.idxmax(), "layer"])
    print("\nbest layer: %d (mean LOTO AUROC %.3f)" % (best, sweep.mean_auroc.max()))
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt.figure(figsize=(5, 3))
        plt.plot(sweep.layer, sweep.mean_auroc, marker="o")
        plt.xlabel("layer")
        plt.ylabel("mean leave-one-topic-out AUROC")
        plt.tight_layout()
        plt.savefig(RESULTS / f"probe_layer_sweep_{tag}.png", dpi=200)
    except ImportError:
        print("(matplotlib not installed: CSV saved, plot skipped. pip install matplotlib)")

    X = np.asarray(H[:, best, :])
    grid = []
    combos = [("mlp", 1.0)] + [("logreg", c) for c in P["classifier_grid"]["logreg_C"]]
    for clip in tqdm(P["clip_percentiles"], desc="clipping x classifier at layer %d" % best):
        for kind, C in combos:
            r = loto(X, y, topics, clip=clip, kind=kind, C=C)
            grid.append(dict(layer=best, clip=clip, kind=kind, C=C, mean_auroc=float(np.mean(list(r.values()))), **r))
    pd.DataFrame(grid).to_csv(RESULTS / f"probe_grid_{tag}.csv", index=False)
    top = max(grid, key=lambda r: r["mean_auroc"])
    print(pd.DataFrame(grid).round(3).to_string(index=False))
    topics_list = sorted(set(topics))
    choice = dict(model=a.model, layer=best, clip=top["clip"], kind=top["kind"], C=top["C"],
                  mean_auroc=top["mean_auroc"], per_topic={t: top[t] for t in topics_list})
    json.dump(choice, open(RESULTS / f"probe_choice_{tag}.json", "w"), indent=2)
    print("\nchosen:", {k: choice[k] for k in ("layer", "clip", "kind", "C", "mean_auroc")})
    print("Tell the team and set probe.layer: %d in config.yaml." % best)


def cmd_final(a):
    tag = model_tag(a.model)
    choice = json.load(open(RESULTS / f"probe_choice_{tag}.json"))
    L = choice["layer"]
    tf = load_truefalse()
    X = np.asarray(np.load(CACHE / f"hidden_{tag}_truefalse.npy", mmap_mode="r")[:, L, :])
    probe = Probe(choice["clip"], choice["kind"], choice["C"]).fit(X, tf.label_true.values)
    noclip = Probe(None, choice["kind"], choice["C"]).fit(X, tf.label_true.values)
    joblib.dump(dict(layer=L, probe=probe, noclip=noclip, choice=choice), CACHE / f"probe_{tag}.joblib")
    print("final probes trained on %d statements, layer %d" % (len(tf), L))

    ids = np.load(CACHE / f"hidden_{tag}_claims_ids.npy")
    Xc = np.asarray(np.load(CACHE / f"hidden_{tag}_claims.npy", mmap_mode="r")[:, L, :])
    out = pd.DataFrame(dict(claim_id=ids, p_know=probe.p_true(Xc), p_know_noclip=noclip.p_true(Xc)))
    main = tag == model_tag(CFG["models"]["probe"])
    path = CACHE / ("scores_pathb.parquet" if main else f"scores_pathb_{tag}.parquet")
    out.to_parquet(path, index=False)
    print("saved", path)

    cl = load_claims()[["claim_id", "dataset", "split", "label"]].merge(out, on="claim_id")
    v = cl[cl.split == "validation"]
    print("sanity check, AUROC of 1 - p_know on the VALIDATION split (not the test split):")
    for ds, g in v.groupby("dataset"):
        if g.label.nunique() == 2:
            print("  %-10s %.3f" % (ds, roc_auc_score(g.label, 1 - g.p_know)))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=["extract", "sweep", "final", "all"])
    ap.add_argument("--model", default=CFG["models"]["probe"])
    ap.add_argument("--only", choices=["truefalse", "claims", "both"], default="both")
    ap.add_argument("--batch-size", type=int, default=P["batch_size"])
    a = ap.parse_args()
    if a.command in ("extract", "all"):
        cmd_extract(a)
    if a.command in ("sweep", "all"):
        cmd_sweep(a)
    if a.command in ("final", "all"):
        cmd_final(a)
