"""Shared helpers for the Phase 2 code: config, paths, claims, deterministic sharding."""
import hashlib
import random
import socket
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parent
CFG = yaml.safe_load(open(ROOT / "config.yaml", encoding="utf-8"))
CACHE = ROOT / CFG["project"]["paths"]["cache"]
RESULTS = ROOT / CFG["project"]["paths"]["results"]
SEED = CFG["project"]["seed"]
CACHE.mkdir(parents=True, exist_ok=True)
RESULTS.mkdir(parents=True, exist_ok=True)


def load_claims():
    return pd.read_parquet(CACHE / "claims.parquet")


def load_truefalse():
    return pd.read_parquet(CACHE / "truefalse.parquet")


def pair_key(query, evidence):
    return hashlib.sha1((query + "\x00" + evidence).encode("utf-8")).hexdigest()[:16]


def shard_tag(spec):
    return spec.replace("/", "of")


def host():
    return socket.gethostname()


def take_shard(items, spec="1/1", weights=None):
    """Deterministic, disjoint shards that together cover every item.
    spec 'i/N' (1-based). weights 'w1,w2,..,wN' (optional) give shard sizes in proportion,
    e.g. --shard 1/3 --weights 55,25,20. Items are shuffled with seed 42 first, so every
    shard gets a mix of datasets and every laptop computes the same split."""
    i, n = (int(x) for x in spec.split("/"))
    assert 1 <= i <= n, "shard must look like i/N with 1 <= i <= N"
    w = [float(x) for x in weights.split(",")] if weights else [1.0] * n
    assert len(w) == n, "--weights needs exactly N numbers"
    order = sorted(items)
    random.Random(SEED).shuffle(order)
    total, acc, cuts = sum(w), 0.0, [0]
    for x in w:
        acc += x
        cuts.append(round(len(order) * acc / total))
    return order[cuts[i - 1]:cuts[i]]
