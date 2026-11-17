"""Mitigation check (Michelle): worked examples of the three flag types plus a small correction / harm check.
  python pipeline/mitigate.py                      examples + rates, about 60 Groq calls on the 120b model
  python pipeline/mitigate.py --n 30               fewer calls for the rates
  python pipeline/mitigate.py --examples-only      one call (the Flag A example)

Uses the frozen thresholds (results/thresholds.json) and the cached scores (cache/scores.parquet) on the TEST split.
Mitigation by flag type (Report 1, section IV-G):
  Flag A (trusted evidence does not support the claim)  -> regenerate the answer with the reference in the prompt (Groq)
  Flag B (untrusted evidence and low internal belief)   -> hedge / abstain (template, no API call)
  Flag D (evidence supports it, the probe disagrees)    -> return the claim with a warning that the source may be unreliable (template)
Rates (HaluEval only, Flag A, claims whose gold answer is not a trivial yes/no):
  correction rate = flagged HALLUCINATED claims whose regenerated answer contains the gold answer
  harm rate       = flagged CORRECT claims whose regenerated answer no longer contains the gold answer
Needs: GROQ_API_KEY in .env, datasets/halueval/data/qa_data.json (from datasets.zip), cache/claims.parquet, cache/scores.parquet.
Writes results/mitigation_examples.md, results/mitigation_check.csv and results/mitigation_rates.json.
Every Groq call is cached by llm.py, so a rerun costs nothing.
"""
import argparse
import json
import math
import random
import re
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common import CACHE, CFG, RESULTS, ROOT, SEED, load_claims  # noqa: E402
from pipeline.diagnose import Ours, diagnose  # noqa: E402

QA_FILE = ROOT / CFG["project"]["paths"]["datasets"] / "halueval" / "data" / "qa_data.json"
SYSTEM = ("You answer a question using ONLY the reference passage. Reply with one complete sentence that states the "
          "answer. If the passage does not contain the answer, say that it does not.")


def norm(s):
    """Lower case, plain hyphens and apostrophes, single spaces (the LLM writes non-breaking hyphens and curly quotes)."""
    for a, b in (("\u2011", "-"), ("\u2010", "-"), ("\u2013", "-"), ("\u2019", "'"), ("\u2018", "'")):
        s = s.replace(a, b)
    return " ".join(s.lower().split())


def contains(text, gold):
    return bool(re.search(r"(?<!\w)" + re.escape(norm(gold)) + r"(?!\w)", norm(text)))


def nontrivial(gold):
    g = (gold or "").strip()
    return len(g) >= 3 and g.lower() not in {"yes", "no"}


def load_gold():
    rows = [json.loads(line) for line in open(QA_FILE, encoding="utf-8") if line.strip()]
    return {"halueval-%d" % i: r["right_answer"] for i, r in enumerate(rows)}


def wilson(k, n, z=1.96):
    if n == 0:
        return float("nan"), float("nan")
    p, d = k / n, 1 + z * z / n
    c, m = p + z * z / (2 * n), z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (c - m) / d, (c + m) / d


def flagged_frame():
    """Test-split claims with the flag type ('A', 'B', 'D' or '') under the frozen thresholds of ours."""
    thr = json.load(open(RESULTS / "thresholds.json"))["ours"]
    df = load_claims().merge(pd.read_parquet(CACHE / "scores.parquet"), on="claim_id", how="left")
    df = df[df.split == "test"]
    m, parts = Ours(), []
    for ds in df.dataset.unique():
        p = thr[CFG["thresholds"]["controlled_set_uses"] if ds == "controlled" else ds]
        d = m.prep(df[df.dataset == ds])
        d = d[m.usable(d)].copy()
        d["kind"] = diagnose(d, p["tau_g"], p["tau_A"], p["tau_B"])[2]
        d["tau_g"] = p["tau_g"]
        parts.append(d)
    return pd.concat(parts, ignore_index=True)


def regenerate(query, evidence):
    from llm import chat

    prompt = "Reference passage:\n%s\n\nQuestion: %s\nAnswer:" % (evidence[:6000], query)
    out = chat(prompt, role="rewrite", system=SYSTEM, temperature=CFG["llm"]["temperature_mitigation"], max_tokens=300)
    return " ".join(out.split())


def hedge(claim):
    return ("I could not check this against a reliable source, and the model does not believe it, so I am not stating it "
            "as fact. (Claim withheld: \"%s\")" % claim)


def warn(claim):
    return ("%s [Warning: the reference supports this claim, but the model's own check disagrees, so the source may be "
            "unreliable. Please verify it independently.]" % claim)


def write_examples(df, gold):
    picks = {}
    a = df[(df.dataset == "halueval") & (df.kind == "A") & (df.label == 1)]
    a = a[a.example_id.map(lambda e: nontrivial(gold.get(e)))].sort_values("claim_id")
    if len(a):
        picks["A"] = (a.iloc[0], "")
    d = df[(df.kind == "D") & (df.dataset == "controlled") & (df.setting == "unreliable")]
    d = d if len(d) else df[df.kind == "D"]
    if len(d):
        picks["D"] = (d.sort_values("claim_id").iloc[0], "")
    b = df[df.kind == "B"]
    note = ""
    if not len(b):
        b, note = df[df.g < df.tau_g], "ILLUSTRATIVE: no claim in the test split is flagged B under the frozen thresholds; this is the untrusted claim with the highest Path B risk."
    if len(b):
        picks["B"] = (b.sort_values("s_B", ascending=False).iloc[0], note)
    names = {"A": "regenerate with the reference", "B": "hedge or abstain", "D": "warn that the source may be unreliable"}
    out = ["# Worked mitigation examples (test split, frozen thresholds)\n"]
    for k in ("A", "B", "D"):
        if k not in picks:
            out.append("## Flag %s (%s)\n\nNo example found in the test split.\n" % (k, names[k]))
            continue
        r, note = picks[k]
        new = regenerate(r["query"], r["evidence"]) if k == "A" else (hedge(r["claim_text"]) if k == "B" else warn(r["claim_text"]))
        out.append("## Flag %s: %s\n" % (k, names[k]))
        if note:
            out.append("*%s*\n" % note)
        out.append("- dataset / setting: %s / %s, true label: %s (1 = hallucinated; for the controlled set 1 = unreliable reference)" % (r["dataset"], r["setting"], int(r["label"])))
        out.append("- scores: g = %.2f, Path A risk = %.2f, Path B risk = %.2f" % (r["g"], r["s_A"], r["s_B"]))
        out.append("- question: %s" % r["query"])
        out.append("- reference (first 400 characters): %s" % " ".join(r["evidence"][:400].split()))
        out.append("- original claim: %s" % r["claim_text"])
        if k == "A":
            out.append("- gold answer: %s" % gold.get(r["example_id"]))
        out.append("- mitigated output: %s\n" % new)
    (RESULTS / "mitigation_examples.md").write_text("\n".join(out), encoding="utf-8")
    print("\n".join(out))
    print("saved", RESULTS / "mitigation_examples.md")


def run_rates(df, gold, n):
    pool = df[(df.dataset == "halueval") & (df.kind == "A")]
    pool = pool[pool.example_id.map(lambda e: nontrivial(gold.get(e)))]
    print("\nFlag A claims on HaluEval test with a non-trivial gold answer: %d hallucinated, %d correct"
          % ((pool.label == 1).sum(), (pool.label == 0).sum()))
    rng, ids = random.Random(SEED), []
    for lab in (1, 0):
        c = sorted(pool[pool.label == lab].claim_id)
        rng.shuffle(c)
        ids += c[: n // 2]
    rows = []
    for _, r in pool[pool.claim_id.isin(ids)].sort_values("claim_id").iterrows():
        try:
            new = regenerate(r["query"], r["evidence"])
        except RuntimeError as e:
            print("\nStopped:", e, "\nRun the same command again later; finished calls are cached.")
            break
        g = gold[r["example_id"]]
        rows.append(dict(claim_id=r["claim_id"], label=int(r["label"]), claim_text=r["claim_text"], gold=g,
                         regenerated=new, contains_gold=contains(new, g)))
    res = pd.DataFrame(rows)
    res.to_csv(RESULTS / "mitigation_check.csv", index=False)
    h, c = res[res.label == 1], res[res.label == 0]
    k1, k0 = int(h.contains_gold.sum()), int((~c.contains_gold).sum())
    out = dict(n_flagged_hallucinated_tested=len(h), corrected=k1, correction_rate=k1 / len(h) if len(h) else None,
               correction_ci95=wilson(k1, len(h)), n_flagged_correct_tested=len(c), harmed=k0,
               harm_rate=k0 / len(c) if len(c) else None, harm_ci95=wilson(k0, len(c)),
               note="HaluEval test split, Flag A only, gold answers that are not yes/no; the regenerated answer is checked for the gold answer string.")
    json.dump(out, open(RESULTS / "mitigation_rates.json", "w"), indent=2)
    print("\ncorrection rate (flagged hallucinated claims fixed): %d / %d = %.2f (95%% CI %.2f to %.2f)"
          % (k1, len(h), out["correction_rate"] or float("nan"), *out["correction_ci95"]))
    print("harm rate (flagged correct claims altered):          %d / %d = %.2f (95%% CI %.2f to %.2f)"
          % (k0, len(c), out["harm_rate"] or float("nan"), *out["harm_ci95"]))
    print("saved", RESULTS / "mitigation_rates.json", "and", RESULTS / "mitigation_check.csv")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=60, help="HaluEval flagged claims for the rates (half hallucinated, half correct)")
    ap.add_argument("--examples-only", action="store_true")
    a = ap.parse_args()
    if not (RESULTS / "thresholds.json").exists():
        sys.exit("results/thresholds.json not found. git pull first.")
    if not QA_FILE.exists():
        sys.exit("%s not found. Unzip datasets.zip into the project root." % QA_FILE)
    frame, gold_answers = flagged_frame(), load_gold()
    print("flag counts on the test split (none = not flagged):", frame.assign(kind=frame.kind.replace("", "none")).groupby("dataset").kind.value_counts().unstack(fill_value=0).to_string(), sep="\n")
    write_examples(frame, gold_answers)
    if not a.examples_only:
        run_rates(frame, gold_answers, a.n)
