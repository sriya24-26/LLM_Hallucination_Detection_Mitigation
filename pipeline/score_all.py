"""Sharded scoring of Path A and the gate, and the merge into cache/scores.parquet.
  python pipeline/score_all.py --part a    --shard 1/3 --weights 55,25,20    Path A shard (roberta-large-mnli)
  python pipeline/score_all.py --part gate --shard 1/3 --weights 55,25,20    gate shard (needs cache/gate.joblib)
  python pipeline/score_all.py --merge                                       merge every shard file in cache/

Shards are deterministic, so the same command with the same shard number always gives the same claims.
Path A is resumable: run the same command again after a crash and it continues where it stopped.
Each run writes cache/latency_<part>_<host>.json (seconds per claim and per NLI pair) for the latency table.
"""
import argparse
import json
import platform
import sys
import time
from pathlib import Path

import pandas as pd
import psutil

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common import CACHE, host, load_claims, pair_key, shard_tag, take_shard  # noqa: E402

CHUNK = 128


def _timing_hint(pairs):
    f = CACHE / ("timing_%s.json" % host())
    if f.exists():
        s = json.loads(f.read_text())["nli_s_per_pair"]
        print("  at your measured %.2f s/pair that is about %.1f hours" % (s, pairs * s / 3600))


def _write_latency(part, items, pairs, seconds):
    info = dict(part=part, host=host(), processor=platform.processor(), cores=psutil.cpu_count(),
                ram_gb=round(psutil.virtual_memory().total / 1e9, 1), items=items, nli_pairs=pairs,
                seconds=round(seconds, 1), ms_per_item=round(1000 * seconds / max(items, 1), 1),
                ms_per_pair=round(1000 * seconds / max(pairs, 1), 1))
    (CACHE / ("latency_%s_%s.json" % (part, host()))).write_text(json.dumps(info, indent=2))


def run_a(a):
    from models.nli import NLI

    claims = load_claims().set_index("claim_id")
    ids = take_shard(claims.index.tolist(), a.shard, a.weights)
    out = CACHE / ("scores_a_shard%s.parquet" % shard_tag(a.shard))
    old = pd.read_parquet(out) if out.exists() else None
    done = set(old.claim_id) if old is not None else set()
    todo = [i for i in ids if i not in done]
    print("shard %s: %d claims, %d already done, %d to do" % (a.shard, len(ids), len(done), len(todo)))
    if not todo:
        return
    nli = NLI()
    sub = claims.loc[todo]
    pairs = sum(len(nli.windows(e)) for e in sub.evidence)
    print("NLI pairs to score: %d" % pairs)
    _timing_hint(pairs)
    frames = [old] if old is not None else []
    t0, n_pairs = time.perf_counter(), 0
    for s in range(0, len(todo), CHUNK):
        c = sub.iloc[s:s + CHUNK]
        ae, ac, nw = nli.path_a(c.evidence.tolist(), c.claim_text.tolist())
        frames.append(pd.DataFrame(dict(claim_id=c.index, a_entail=ae, a_contra=ac, n_windows=nw)))
        pd.concat(frames, ignore_index=True).to_parquet(out, index=False)
        n_pairs += int(nw.sum())
        print("  %d / %d claims  (%.0f min elapsed)" % (min(s + CHUNK, len(todo)), len(todo), (time.perf_counter() - t0) / 60))
    _write_latency("path_a", len(todo), n_pairs, time.perf_counter() - t0)
    print("saved", out)


def run_gate(a):
    from models.gate import gate_features, load_embedder, load_gate
    from models.nli import NLI

    claims = load_claims()
    pairs = claims.drop_duplicates(["query", "evidence"])[["query", "evidence"]].copy()
    pairs["pair_key"] = [pair_key(q, e) for q, e in zip(pairs["query"], pairs.evidence)]
    pairs = pairs.set_index("pair_key")
    keys = take_shard(pairs.index.tolist(), a.shard, a.weights)
    print("shard %s: %d unique (query, passage) pairs of %d" % (a.shard, len(keys), len(pairs)))
    nli, emb, clf = NLI(), load_embedder(), load_gate()
    out = CACHE / ("scores_gate_shard%s.parquet" % shard_tag(a.shard))
    frames, t0, n_pairs = [], time.perf_counter(), 0
    for s in range(0, len(keys), 100):
        k = keys[s:s + 100]
        sub = pairs.loc[k]
        X = gate_features(nli, emb, sub["query"].tolist(), sub.evidence.tolist())
        frames.append(pd.DataFrame(dict(pair_key=k, f_nli=X[:, 0], f_cos=X[:, 1], g=clf.predict_proba(X)[:, 1])))
        pd.concat(frames, ignore_index=True).to_parquet(out, index=False)
        n_pairs += sum(len(nli.windows(e, 3)) for e in sub.evidence)
        print("  %d / %d pairs  (%.0f min elapsed)" % (min(s + 100, len(keys)), len(keys), (time.perf_counter() - t0) / 60))
    _write_latency("gate", len(keys), n_pairs, time.perf_counter() - t0)
    print("saved", out)


def _read_all(pattern, key):
    files = sorted(CACHE.glob(pattern))
    if not files:
        return None
    return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True).drop_duplicates(key, keep="last")


def merge():
    claims = load_claims()[["claim_id", "query", "evidence"]]
    claims["pair_key"] = [pair_key(q, e) for q, e in zip(claims["query"], claims.evidence)]
    out = claims[["claim_id", "pair_key"]]
    a = _read_all("scores_a_shard*.parquet", "claim_id")
    g = _read_all("scores_gate_shard*.parquet", "pair_key")
    if a is not None:
        out = out.merge(a, on="claim_id", how="left")
    if g is not None:
        out = out.merge(g[["pair_key", "g"]], on="pair_key", how="left")
    for name, f in [("pathb", "scores_pathb.parquet")]:
        if (CACHE / f).exists():
            out = out.merge(pd.read_parquet(CACHE / f), on="claim_id", how="left")
    sc = _read_all("scores_selfcheck_shard*.parquet", "claim_id")
    if sc is not None:
        out = out.merge(sc[["claim_id", "selfcheck"]], on="claim_id", how="left")
    out = out.drop(columns="pair_key")
    out.to_parquet(CACHE / "scores.parquet", index=False)
    print("saved cache/scores.parquet with %d rows. Missing values per column:" % len(out))
    print(out.isna().sum().to_string())
    print("(selfcheck is expected to be missing for the 351 controlled claims)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", choices=["a", "gate"])
    ap.add_argument("--shard", default="1/1")
    ap.add_argument("--weights", default=None, help="e.g. 55,25,20 (one number per shard)")
    ap.add_argument("--merge", action="store_true")
    a = ap.parse_args()
    if a.merge:
        merge()
    elif a.part == "a":
        run_a(a)
    elif a.part == "gate":
        run_gate(a)
    else:
        ap.error("use --part a, --part gate or --merge")
