"""Loader for the Azaria & Mitchell true-false dataset (six topic files).

Output columns: statement, topic, label_true (1 = true, as released),
                label (1 = hallucinated/false, the project convention).
generated_true_false.csv (LLM-generated sentences) is NOT part of the six topics
and is deliberately excluded from probe training and leave-one-topic-out.
"""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "datasets" / "truefalse"
TOPICS = ["cities", "inventions", "elements", "animals", "companies", "facts"]


def load_truefalse(data_dir: Path = DATA_DIR) -> pd.DataFrame:
    frames = []
    for topic in TOPICS:
        path = data_dir / f"{topic}_true_false.csv"
        df = pd.read_csv(path)
        df = df[["statement", "label"]].dropna()
        df["statement"] = df["statement"].astype(str).str.strip()
        before = len(df)
        df = df.drop_duplicates(subset=["statement", "label"])
        if len(df) != before:
            print("  %-10s dropped %d exact duplicate row(s)" % (topic, before - len(df)))
        df["topic"] = topic
        frames.append(df)
    out = pd.concat(frames, ignore_index=True)
    out = out.rename(columns={"label": "label_true"})
    out["label_true"] = out["label_true"].astype(int)
    out["label"] = 1 - out["label_true"]  # project convention: 1 = hallucinated
    return out


if __name__ == "__main__":
    df = load_truefalse()
    print("total statements:", len(df))
    print(df.groupby("topic").agg(n=("label", "size"), false_share=("label", "mean")).round(3))
