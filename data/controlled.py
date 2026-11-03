"""Controlled unreliable-evidence set (Report 2 plan, Section 4), rebuilt so claim and passage agree.

For each HaluEval QA question whose right-answer string is found in its passage:
    wrong phrase  = short wrong answer extracted by the 120b model (cached)
    unreliable    = passage with the right answer replaced by the wrong phrase + a claim stating that phrase -> label 1
    reliable      = original passage + the right claim                                                       -> label 0
    reliable_wrongclaim = original passage + the same wrong claim                                            -> label 0
Questions are skipped when the right answer is trivial or not found, the wrong phrase equals/contains/is contained
in the right answer or already appears in the original passage (nothing would be corrupted), or the rewritten
claim does not contain the phrase. Same example_id as the HaluEval claims, so the splits match.
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
from halueval import CFG, COLUMNS, HALLU_SYSTEM, build_halueval_claims, load_qa_sample, rewrite  # noqa: E402

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
    claims = build_halueval_claims(sample).set_index("claim_id")  # all cached, no new calls
    rows = []
    skipped = dict(trivial=0, not_found=0, same_answer=0, overlap=0, in_passage=0, claim_mismatch=0)
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
        if len(wrong) < 2 or wrong.lower() == ans.lower():
            skipped["same_answer"] += 1
            continue
        if ans.lower() in wrong.lower() or wrong.lower() in ans.lower():
            skipped["overlap"] += 1
            continue
        if pattern(wrong).search(r["knowledge"]):
            skipped["in_passage"] += 1
            continue
        wrong_claim = rewrite(r["question"], wrong, HALLU_SYSTEM)  # one new call per surviving question
        if not pattern(wrong).search(wrong_claim):
            skipped["claim_mismatch"] += 1
            continue
        edited = pat.sub(lambda m: wrong, r["knowledge"])
        base = dict(dataset="controlled", example_id="halueval-%d" % r["qid"], split=None, query=r["question"])
        rows.append(dict(base, claim_id="controlled-%d-unreliable" % r["qid"], setting="unreliable",
                         evidence=edited, claim_text=wrong_claim, label=1))
        rows.append(dict(base, claim_id="controlled-%d-reliable" % r["qid"], setting="reliable",
                         evidence=r["knowledge"], claim_text=claims.loc["halueval-%d-right" % r["qid"], "claim_text"], label=0))
        rows.append(dict(base, claim_id="controlled-%d-reliable_wrongclaim" % r["qid"], setting="reliable_wrongclaim",
                         evidence=r["knowledge"], claim_text=wrong_claim, label=0))
    print("skipped questions:", skipped)
    return pd.DataFrame(rows, columns=COLUMNS)


if __name__ == "__main__":
    df = build_controlled()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT, index=False)
    print("saved %d pairs -> %s" % (len(df), OUT))
    print("label 1 = unreliable reference:", df.label.value_counts().to_dict())
    for _, r in df.head(3).iterrows():
        print("-", r.setting, "| claim:", r.claim_text)