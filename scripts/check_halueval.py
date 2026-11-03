"""Check the HaluEval label fix: the label-1 claim must state the hallucinated answer, label-0 the right one."""
import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "data"))
from halueval import CFG, load_qa_sample  # noqa: E402


def tok(s):
    return set(re.findall(r"\w+", s.lower()))


def cover(words, claim):
    return len(words & tok(claim)) / len(words)


claims = pd.read_parquet(ROOT / CFG["project"]["paths"]["cache"] / "claims_halueval.parquet").set_index("claim_id")
bad, checked = [], 0
for r in load_qa_sample():
    right = tok(r["right_answer"]) - tok(r["hallucinated_answer"])
    wrong = tok(r["hallucinated_answer"]) - tok(r["right_answer"])
    if not right or not wrong:
        continue
    checked += 1
    h = claims.loc["halueval-%d-halluc" % r["qid"], "claim_text"]
    g = claims.loc["halueval-%d-right" % r["qid"], "claim_text"]
    if cover(right, h) > cover(wrong, h) or cover(wrong, g) > cover(right, g):
        bad.append((r["question"], r["right_answer"], r["hallucinated_answer"], g, h))
print("checked %d, suspicious %d" % (checked, len(bad)))
for q, ra, ha, g, h in bad[:5]:
    print("\nQ:", q, "\n right:", ra, "\n wrong:", ha, "\n label0 claim:", g, "\n label1 claim:", h)