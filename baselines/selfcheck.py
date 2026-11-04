"""SelfCheckGPT baseline (proxy sampler).  Owner: Kriti. Sampling is run by all four members, one shard each.
  python baselines/selfcheck.py sample --shard 1/4                       k samples per prompt from gpt-oss-20b (Groq)
  python baselines/selfcheck.py score  --shard 1/3 --weights 55,25,20    NLI contradiction of each claim vs its samples

One prompt = one source example (dataset|example_id): the evidence (cut to selfcheck.max_evidence_tokens) plus the
query. Samples are collected for sample_id 0 first, then 1, ... so if the daily quota runs out every prompt still
has the same number of samples (the plan allows lowering k to 3). Controlled-set claims are not scored.
Score of a claim = mean contradiction probability (roberta-large-mnli) of the claim against the k samples.
Sample files: cache/selfcheck_samples_<i>of<N>.jsonl. Copy all of them into every laptop's cache/ before `score`.
"""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common import CACHE, CFG, load_claims, shard_tag, take_shard  # noqa: E402
from eval.tune import Method, best_single  # noqa: E402

S = CFG["selfcheck"]


def prompt_table():
    c = load_claims()
    c = c[c.dataset != "controlled"]
    t = c.groupby(["dataset", "example_id"], as_index=False).first()[["dataset", "example_id", "query", "evidence"]]
    t["key"] = t.dataset + "|" + t.example_id
    return t.set_index("key")


def build_prompt(query, evidence):
    ev = evidence[: S["max_evidence_tokens"] * 4]  # about 4 characters per token
    return "Context:\n%s\n\nTask: %s\n\nUsing the context, respond in at most 5 sentences." % (ev, query)


def cmd_sample(a):
    from llm import chat, usage_today

    t = prompt_table()
    keys = take_shard(t.index.tolist(), a.shard, a.weights)
    path = CACHE / ("selfcheck_samples_%s.jsonl" % shard_tag(a.shard))
    done = set()
    if path.exists():
        for line in open(path, encoding="utf-8"):
            r = json.loads(line)
            done.add((r["key"], r["sample_id"]))
    todo = [(k, s) for s in range(S["k_samples"]) for k in keys if (k, s) not in done]
    print("shard %s: %d prompts, %d calls to make (%d already done). Calls used today: %s"
          % (a.shard, len(keys), len(todo), len(done), usage_today()))
    with open(path, "a", encoding="utf-8") as f:
        for key, s in tqdm(todo, desc="sampling"):
            row = t.loc[key]
            try:
                text = chat(build_prompt(row["query"], row.evidence), role="sampler",
                            temperature=S["temperature"], max_tokens=400, sample_id=s)
            except RuntimeError as e:
                print("\nStopped:", e, "\nRun the same command again later; finished samples are kept.")
                return
            f.write(json.dumps(dict(key=key, sample_id=s, text=text), ensure_ascii=False) + "\n")
            f.flush()


def load_samples():
    samples = defaultdict(dict)
    for f in sorted(CACHE.glob("selfcheck_samples_*.jsonl")):
        for line in open(f, encoding="utf-8"):
            r = json.loads(line)
            samples[r["key"]][r["sample_id"]] = r["text"]
    return {k: [v[i] for i in sorted(v)] for k, v in samples.items()}


def cmd_score(a):
    from models.nli import NLI

    samples = load_samples()
    claims = load_claims()
    claims = claims[claims.dataset != "controlled"].copy()
    claims["key"] = claims.dataset + "|" + claims.example_id
    ids = take_shard(claims.claim_id.tolist(), a.shard, a.weights)
    c = claims.set_index("claim_id").loc[ids]
    missing = int((~c.key.isin(samples.keys())).sum())
    print("shard %s: %d claims; %d have no samples yet (they are skipped)" % (a.shard, len(c), missing))
    c = c[c.key.isin(samples.keys())]
    out = CACHE / ("scores_selfcheck_shard%s.parquet" % shard_tag(a.shard))
    old = pd.read_parquet(out) if out.exists() else None
    if old is not None:
        c = c[~c.index.isin(old.claim_id)]
    nli = NLI()
    frames = [old] if old is not None else []
    for s in range(0, len(c), 100):
        ch = c.iloc[s:s + 100]
        prem, hyp, owner = [], [], []
        for i, (cid, r) in enumerate(ch.iterrows()):
            for text in samples[r.key]:
                prem.append(text)
                hyp.append(r.claim_text)
                owner.append(i)
        p = nli.probs(prem, hyp)
        owner = np.array(owner)
        n_s = np.bincount(owner, minlength=len(ch))
        frames.append(pd.DataFrame(dict(claim_id=ch.index, selfcheck=np.bincount(owner, weights=p[:, 0], minlength=len(ch)) / n_s,
                                        n_samples=n_s)))
        pd.concat(frames, ignore_index=True).to_parquet(out, index=False)
        print("  %d / %d claims" % (min(s + 100, len(c)), len(c)))
    print("saved", out)


class SelfCheck(Method):
    name = "selfcheckgpt"
    needs = ["selfcheck"]

    def risk(self, df):
        return df["selfcheck"].values

    def tune(self, v, k):
        return best_single(v["selfcheck"].values, v["label"].values.astype(bool), k)

    def predict(self, df, p):
        return df["selfcheck"].values > p["tau"]


METHOD = SelfCheck()

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=["sample", "score"])
    ap.add_argument("--shard", default="1/1")
    ap.add_argument("--weights", default=None)
    a = ap.parse_args()
    cmd_sample(a) if a.command == "sample" else cmd_score(a)
