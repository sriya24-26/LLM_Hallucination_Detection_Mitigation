"""HaluEval QA loader -> claims (Report 2 plan, Section 3 and 4).

For each sampled question there are two claims:
    right answer        -> label 0 (not hallucinated)
    hallucinated answer -> label 1 (hallucinated)
Each (question, answer) is rewritten by Groq as ONE declarative sentence, because NLI
and the probe need full sentences. Evidence = the HaluEval knowledge passage.
qa_data.json is JSON Lines (one object per line), so it is read line by line.

Output columns follow claims.parquet; `split` is filled in later by data/splits.py.
Run from the project root:  python data/halueval.py
"""
import json
import random
import sys
from pathlib import Path

import pandas as pd
import yaml
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from llm import chat  # noqa: E402

CFG = yaml.safe_load(open(ROOT / "config.yaml", encoding="utf-8"))
QA_FILE = ROOT / CFG["project"]["paths"]["datasets"] / "halueval" / "data" / "qa_data.json"
OUT = ROOT / CFG["project"]["paths"]["cache"] / "claims_halueval.parquet"

SYSTEM = (
    "You convert a question and its answer into ONE complete declarative sentence that "
    "states the answer as a fact. Keep every detail of the answer, add no new facts, "
    "and output only the sentence."
)
HALLU_SYSTEM = (
    "You rewrite a question and a given answer into ONE complete declarative sentence that "
    "states that answer. The answer may be factually WRONG. Do NOT correct it, check it, or "
    "replace it with the true answer: restate it exactly as given, keeping its wrong details. "
    "Output only the sentence."
)
COLUMNS = ["claim_id", "dataset", "setting", "example_id", "split", "query", "evidence", "claim_text", "label"]


def load_qa_sample(n=None, seed=None):
    """Fixed random sample of n questions (seed 42), each tagged with its row index `qid`."""
    n = n or CFG["sampling"]["halueval_questions"]
    seed = seed if seed is not None else CFG["project"]["seed"]
    rows = [json.loads(line) for line in open(QA_FILE, encoding="utf-8") if line.strip()]
    for i, r in enumerate(rows):
        r["qid"] = i
    chosen = random.Random(seed).sample(rows, n)
    return sorted(chosen, key=lambda r: r["qid"])


def rewrite(question, answer, system=SYSTEM):
    out = chat("Question: %s\nAnswer: %s" % (question, answer), role="rewrite",
               system=system, max_tokens=250)
    return " ".join(out.split())


def build_halueval_claims(sample=None):
    sample = sample if sample is not None else load_qa_sample()
    rows = []
    for r in tqdm(sample, desc="rewriting HaluEval claims"):
        base = dict(dataset="halueval", setting="qa", example_id="halueval-%d" % r["qid"],
                    split=None, query=r["question"], evidence=r["knowledge"])
        rows.append(dict(base, claim_id="halueval-%d-right" % r["qid"],
                         claim_text=rewrite(r["question"], r["right_answer"]), label=0))
        rows.append(dict(base, claim_id="halueval-%d-halluc" % r["qid"],
                         claim_text=rewrite(r["question"], r["hallucinated_answer"], HALLU_SYSTEM), label=1))
    return pd.DataFrame(rows, columns=COLUMNS)


if __name__ == "__main__":
    df = build_halueval_claims()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT, index=False)
    print("saved %d claims -> %s" % (len(df), OUT))
    print(df.label.value_counts().to_dict())
    for _, r in df.head(4).iterrows():
        print("-", r.label, "|", r.claim_text)