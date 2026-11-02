"""RAGTruth loader -> sentence-level claims (Report 2 plan, Sections 3 and 4).

Uses the official TEST split, drops low-quality responses, samples `responses_per_task`
responses per task (Summary, QA, Data2txt), splits each response into sentences with pysbd,
and labels a sentence hallucinated (1) if it overlaps any annotated span (character offsets).
Evidence comes from source_info.jsonl: Summary = article text, QA = passages,
Data2txt = the structured business record turned into "key: value" lines.
The gate needs a query: QA has a real question; for Summary and Data2txt we use a fixed
stand-in query describing the task (a design choice, state it in the report).
example_id = the source (several models answer the same source, so they must share a split).
Run from the project root:  python data/ragtruth.py
"""
import json
import sys
from pathlib import Path

import pandas as pd
import pysbd
import yaml

ROOT = Path(__file__).resolve().parent.parent
CFG = yaml.safe_load(open(ROOT / "config.yaml", encoding="utf-8"))
RT = ROOT / CFG["project"]["paths"]["datasets"] / "ragtruth" / "dataset"
OUT = ROOT / CFG["project"]["paths"]["cache"] / "claims_ragtruth.parquet"
COLUMNS = ["claim_id", "dataset", "setting", "example_id", "split", "query", "evidence", "claim_text", "label"]
SEG = pysbd.Segmenter(language="en", clean=False)


def to_text(obj, indent=0):
    """Turn a nested dict/list into readable 'key: value' lines."""
    pad = "  " * indent
    lines = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            if v in (None, "", [], {}):
                continue
            if isinstance(v, (dict, list)):
                lines.append("%s%s:" % (pad, k))
                lines += to_text(v, indent + 1)
            else:
                lines.append("%s%s: %s" % (pad, k, v))
    elif isinstance(obj, list):
        for v in obj:
            lines += to_text(v, indent) if isinstance(v, (dict, list)) else ["%s- %s" % (pad, v)]
    else:
        lines.append("%s%s" % (pad, obj))
    return lines


def evidence_and_query(task, info):
    if task == "QA":
        return info.get("passages", ""), info.get("question", "")
    if task == "Data2txt":
        name = info.get("name", "the business") if isinstance(info, dict) else "the business"
        return "\n".join(to_text(info)), "Write an objective overview of the business %s." % name
    return str(info), "Summarize the following news article."


def sentences_with_offsets(text):
    """Sentences and their (start, end) character offsets, found left to right."""
    out, cursor = [], 0
    for s in SEG.segment(text):
        s = s.strip()
        if not s:
            continue
        start = text.find(s, cursor)
        if start < 0:
            continue
        out.append((s, start, start + len(s)))
        cursor = start + len(s)
    return out


def build_ragtruth_claims():
    seed = CFG["project"]["seed"]
    n = CFG["sampling"]["ragtruth_responses_per_task"]
    drop = CFG["sampling"]["ragtruth_drop_quality"]
    min_words = CFG["sampling"]["ragtruth_min_sentence_words"]

    resp = pd.read_json(RT / "response.jsonl", lines=True)
    src = pd.read_json(RT / "source_info.jsonl", lines=True)
    df = resp.merge(src[["source_id", "task_type", "source_info"]], on="source_id")
    df = df[(df.split == "test") & (~df.quality.isin(drop))]

    rows = []
    for task, grp in df.groupby("task_type"):
        grp = grp.sample(n=min(n, len(grp)), random_state=seed)
        for _, r in grp.iterrows():
            evidence, query = evidence_and_query(task, r.source_info)
            spans = [(int(l["start"]), int(l["end"])) for l in (r.labels or [])]
            for i, (sent, a, b) in enumerate(sentences_with_offsets(r.response)):
                if len(sent.split()) < min_words:
                    continue
                hallu = int(any(s < b and e > a for s, e in spans))
                rows.append(dict(claim_id="ragtruth-%d-s%d" % (r.id, i), dataset="ragtruth",
                                 setting=task.lower(), example_id="ragtruth-%d" % r.source_id,
                                 split=None, query=query, evidence=evidence, claim_text=sent, label=hallu))
    return pd.DataFrame(rows, columns=COLUMNS)


if __name__ == "__main__":
    out = build_ragtruth_claims()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(OUT, index=False)
    print("saved %d sentences -> %s" % (len(out), OUT))
    print(out.groupby("setting").agg(sentences=("label", "size"), hallucinated_share=("label", "mean")).round(3))
    print("target is about 900 sentences in total: raise or lower sampling.ragtruth_responses_per_task in config.yaml")
