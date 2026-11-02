"""Controlled unreliable-evidence set (Report 2 plan, Section 4).
 
For each HaluEval QA question whose right-answer string is found in its passage:
    unreliable pair: passage with the right answer replaced by the (short) hallucinated
                     answer, paired with the hallucinated claim   -> label 1
    reliable pair:   original passage paired with the right claim -> label 0
Ground truth is exact because we made the corruption. Questions whose right answer is
not found (or is a trivial yes/no) are skipped, so the final count is below 2 x n.
 
HaluEval's hallucinated_answer is often a full sentence, so one extra Groq call per
question extracts the short answer phrase to substitute into the passage.
This reuses the same sampled questions and the same example_id as the HaluEval claims,
so a question lands in the same validation/test split in both sets.
Run from the project root:  python data/controlled.py
"""
import re
import sys
from pathlib import Path
 
import pandas as pd
from tqdm import tqdm
 
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "data"))
from llm import chat  # noqa: E402
from halueval import CFG, COLUMNS, build_halueval_claims, load_qa_sample  # noqa: E402
 
OUT = ROOT / CFG["project"]["paths"]["cache"] / "claims_controlled.parquet"
SHORT_SYSTEM = (
    "Given a question and a wrong answer, output ONLY the shortest phrase that answers the "
    "question (a name, number, date or place), with no extra words and no full sentence."
)
 
 
def pattern(ans):
    return re.compile(r"(?<!\w)" + re.escape(ans.strip()) + r"(?!\w)", re.IGNORECASE)
 
 
def short_wrong(question, wrong_answer):
    out = chat("Question: %s\nWrong answer: %s" % (question, wrong_answer), role="rewrite",
               system=SHORT_SYSTEM, max_tokens=250)
    return " ".join(out.split()).strip(" .\"'")
 
 
def build_controlled():
    sample = load_qa_sample()
    claims = build_halueval_claims(sample).set_index("claim_id")
    rows, skipped = [], {"trivial": 0, "not_found": 0, "same_answer": 0}
    for r in tqdm(sample, desc="building controlled pairs"):
        ans = r["right_answer"].strip()
        if len(ans) < 3 or ans.lower() in {"yes", "no"}:
            skipped["trivial"] += 1
            continue
        pat = pattern(ans)
        if not pat.search(r["knowledge"]):
            skipped["not_found"] += 1
            continue
        wrong = short_wrong(r["question"], r["hallucinated_answer"])
        if not wrong or wrong.lower() == ans.lower():
            skipped["same_answer"] += 1
            continue
        edited = pat.sub(lambda m: wrong, r["knowledge"])
        base = dict(dataset="controlled", example_id="halueval-%d" % r["qid"], split=None, query=r["question"])
        rows.append(dict(base, claim_id="controlled-%d-unreliable" % r["qid"], setting="unreliable",
                         evidence=edited, claim_text=claims.loc["halueval-%d-halluc" % r["qid"], "claim_text"], label=1))
        rows.append(dict(base, claim_id="controlled-%d-reliable" % r["qid"], setting="reliable",
                         evidence=r["knowledge"], claim_text=claims.loc["halueval-%d-right" % r["qid"], "claim_text"], label=0))
    print("skipped questions:", skipped)
    return pd.DataFrame(rows, columns=COLUMNS)
 
 
if __name__ == "__main__":
    df = build_controlled()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT, index=False)
    print("saved %d pairs -> %s" % (len(df), OUT))
    print("label 1 = unreliable reference:", df.label.value_counts().to_dict())
    for _, r in df.head(2).iterrows():
        print("-", r.setting, "| claim:", r.claim_text)
 