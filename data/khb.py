"""KnowHalBench (RefChecker) loader -> claims (Report 2 plan, Sections 3 and 4).

Claim = one annotated triplet (head, relation, tail) joined into text.
Label: Entailment -> 0 (not hallucinated); Contradiction and Neutral -> 1 (hallucinated).
Evidence = the reference passages for that question (Zero: NQ long answer, Noisy: MS MARCO
passages, Accurate: the Dolly context). These are NOT shipped with the repo: build them first
with benchmark/data/download_data.sh (Colab) so that
    datasets/refchecker/benchmark/data/{zero_context/nq.json, noisy_context/msmarco.json,
                                        accurate_context/dolly.json}
exist. A setting whose evidence file is missing is skipped with a warning.
example_id = the question (7 models answer each question, so they share a split).
Run from the project root:  python data/khb.py
"""
import json
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parent.parent
CFG = yaml.safe_load(open(ROOT / "config.yaml", encoding="utf-8"))
BENCH = ROOT / CFG["project"]["paths"]["datasets"] / "refchecker" / "benchmark"
OUT = ROOT / CFG["project"]["paths"]["cache"] / "claims_khb.parquet"
COLUMNS = ["claim_id", "dataset", "setting", "example_id", "split", "query", "evidence", "claim_text", "label"]
# setting name -> (folder, file prefix, evidence file)
SETTINGS = {
    "zero": ("zero_context", "nq", "nq.json"),
    "noisy": ("noisy_context", "msmarco", "msmarco.json"),
    "accurate": ("accurate_context", "dolly", "dolly.json"),
}


def triplet_text(t):
    parts = [str(x).strip().strip('"') for x in t if str(x).strip()]
    return " ".join(parts)


def build_khb_claims():
    seed = CFG["project"]["seed"]
    per_setting = CFG["sampling"]["khb_claims_per_setting"]
    frames = []
    for setting, (folder, prefix, ctx_file) in SETTINGS.items():
        ctx_path = BENCH / "data" / folder / ctx_file
        if not ctx_path.exists():
            print("WARNING: %s missing, skipping the %s setting (build it with download_data.sh)" % (ctx_path, setting))
            continue
        ctx = {str(d["id"]): d for d in json.load(open(ctx_path, encoding="utf-8"))}
        rows = []
        for f in sorted((BENCH / "human_annotations_v1" / folder).glob("%s_*_answers.json" % prefix)):
            model = f.stem[len(prefix) + 1:-len("_answers")]
            for item in json.load(open(f, encoding="utf-8")):
                qid = str(item["id"])
                if qid not in ctx:
                    continue
                evidence = "\n\n".join(c for c in ctx[qid]["context"] if c)
                for k, tr in enumerate(item.get("claude2_response_kg") or []):
                    label = tr.get("human_label")
                    if label not in ("Entailment", "Neutral", "Contradiction") or not triplet_text(tr["triplet"]):
                        continue
                    rows.append(dict(claim_id="khb-%s-%s-%s-%d" % (setting, qid, model, k), dataset="khb",
                                     setting=setting, example_id="khb-%s-%s" % (setting, qid), split=None,
                                     query=ctx[qid]["question"], evidence=evidence,
                                     claim_text=triplet_text(tr["triplet"]), label=int(label != "Entailment")))
        df = pd.DataFrame(rows, columns=COLUMNS)
        if len(df) > per_setting:  # stratified by label, keeps the original proportions
            df = df.groupby("label", group_keys=False).sample(frac=per_setting / len(df), random_state=seed)
        frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=COLUMNS)


if __name__ == "__main__":
    out = build_khb_claims()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(OUT, index=False)
    print("saved %d claims -> %s" % (len(out), OUT))
    print(out.groupby("setting").agg(claims=("label", "size"), hallucinated_share=("label", "mean")).round(3))
    print("sample claims:")
    for _, r in out.head(5).iterrows():
        print("-", r.label, "|", r.claim_text)
