"""Reliability gate: logistic regression on two features -> g in [0, 1].  Owner: Erina.
  python models/gate.py train [--eval-test]

Features per (query, passage):
  1. max entailment of "This passage answers the question: {query}" over the first gate.max_windows windows
     of the passage (roberta-large-mnli, zero-shot)
  2. MiniLM cosine similarity of query and passage
Training pairs come from the VALIDATION split only: a question's own passage = relevant, another question's
passage (same dataset and setting) = not relevant. Only datasets with a real question are used (HaluEval,
KnowHalBench, RAGTruth QA); the RAGTruth Summary and Data2txt queries are fixed stand-ins.
Output: cache/gate.joblib (share it on Drive, every laptop needs the same file).
"""
import argparse
import random
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common import CACHE, CFG, SEED, load_claims  # noqa: E402

G = CFG["gate"]


def load_embedder():
    from sentence_transformers import SentenceTransformer

    m = SentenceTransformer(CFG["models"]["embedding"], device="cpu")
    m.max_seq_length = G["embedding_max_tokens"]
    return m


def gate_features(nli, emb, queries, passages, batch_size=16):
    """(n, 2) array: [NLI relevance score, MiniLM cosine]."""
    prem, hyp, owner = [], [], []
    for k, (q, p) in enumerate(zip(queries, passages)):
        for w in nli.windows(p, G["max_windows"]):
            prem.append(w)
            hyp.append(G["hypothesis_template"].format(query=q))
            owner.append(k)
    pr = nli.probs(prem, hyp, batch_size)
    f_nli = np.zeros(len(queries), np.float32)
    np.maximum.at(f_nli, np.array(owner), pr[:, 2])
    qe = emb.encode(list(queries), batch_size=64, normalize_embeddings=True, show_progress_bar=False)
    pe = emb.encode(list(passages), batch_size=32, normalize_embeddings=True, show_progress_bar=False)
    return np.stack([f_nli, (qe * pe).sum(1)], axis=1)


def relevance_pairs(claims, split):
    d = claims[(claims.split == split) & (claims.dataset.isin(["halueval", "khb"]) |
                                          ((claims.dataset == "ragtruth") & (claims.setting == "qa")))]
    ex = d.groupby(["dataset", "setting", "example_id"], as_index=False).first()
    rng = random.Random(SEED)
    rows = []
    for _, grp in ex.groupby(["dataset", "setting"]):
        recs = grp.to_dict("records")
        if len(recs) < 2:
            continue
        for r in recs:
            rows.append((r["query"], r["evidence"], 1))
            for _ in range(20):
                o = rng.choice(recs)
                if o["example_id"] != r["example_id"] and o["evidence"] != r["evidence"]:
                    rows.append((r["query"], o["evidence"], 0))
                    break
    return pd.DataFrame(rows, columns=["query", "passage", "relevant"])


def load_gate():
    return joblib.load(CACHE / "gate.joblib")["clf"]


def train(eval_test=False):
    from models.nli import NLI

    claims, nli, emb = load_claims(), NLI(), load_embedder()
    pairs = relevance_pairs(claims, "validation")
    print("training pairs: %d (%d relevant, %d not)" % (len(pairs), pairs.relevant.sum(), (1 - pairs.relevant).sum()))
    X = gate_features(nli, emb, pairs["query"].tolist(), pairs.passage.tolist())
    y = pairs.relevant.values
    clf = LogisticRegression(C=G["logreg_C"], max_iter=1000, random_state=SEED).fit(X, y)
    print("coefficients [nli, cosine]:", clf.coef_.round(2).tolist(), "intercept:", clf.intercept_.round(2).tolist())
    print("train AUROC: %.3f" % roc_auc_score(y, clf.predict_proba(X)[:, 1]))
    joblib.dump({"clf": clf}, CACHE / "gate.joblib")
    print("saved", CACHE / "gate.joblib")
    if eval_test:
        t = relevance_pairs(claims, "test")
        Xt = gate_features(nli, emb, t["query"].tolist(), t.passage.tolist())
        print("test-split relevance AUROC: %.3f (%d pairs)" % (roc_auc_score(t.relevant.values, clf.predict_proba(Xt)[:, 1]), len(t)))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=["train"])
    ap.add_argument("--eval-test", action="store_true", help="also report relevance AUROC on test-split pairs (extra NLI time)")
    a = ap.parse_args()
    train(a.eval_test)
