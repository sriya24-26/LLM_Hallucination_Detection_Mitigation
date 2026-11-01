"""Phase 0, step 4: inspect the four raw datasets.

Run from the project root:  python scripts/inspect_data.py
Expected layout (git-ignored):
    datasets/truefalse/   six topic CSVs (cities, inventions, elements, animals, companies, facts)
    datasets/refchecker/  clone of amazonscience/RefChecker (benchmark files inside)
    datasets/ragtruth/    clone of ParticleMedia/RAGTruth (dataset/ folder inside)
    datasets/halueval/    clone of RUCAIBox/HaluEval (data/qa_data.json inside)

For every data file it prints: record count, fields, 3 sample rows, label balance.
Use the output to correct any wrong assumption in the plan BEFORE writing loaders.
"""
import json
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "datasets"
EXTS = {".csv", ".tsv", ".json", ".jsonl"}
SKIP_DIRS = {".git", "__pycache__", "node_modules", "baseline", "image", "assets",
             "analysis", "evaluation", "generation", ".idea"}
LABEL_HINTS = ("label", "hallucinat", "truth", "true", "factual")
MAX_FILES = 12  # per dataset, so big repos do not flood the screen


def short(v, n=140):
    s = json.dumps(v, ensure_ascii=False) if isinstance(v, (dict, list)) else str(v)
    return s if len(s) <= n else s[:n] + "..."


def load(path: Path):
    """Return a list of records (dicts) or None if unreadable."""
    try:
        if path.suffix in {".csv", ".tsv"}:
            sep = "\t" if path.suffix == ".tsv" else ","
            df = pd.read_csv(path, sep=sep, on_bad_lines="skip")
            return df.to_dict("records")
        text = path.read_text(encoding="utf-8")
        try:
            obj = json.loads(text)
            if isinstance(obj, list):
                return obj
            if isinstance(obj, dict):
                # dict of lists/records -> try to find the main list
                for v in obj.values():
                    if isinstance(v, list) and v and isinstance(v[0], dict):
                        return v
                return [obj]
        except json.JSONDecodeError:
            return [json.loads(line) for line in text.splitlines() if line.strip()]
    except Exception as e:  # noqa: BLE001
        print("   !! could not read: %s" % e)
    return None


def label_balance(records):
    keys = set()
    for r in records[:50]:
        if isinstance(r, dict):
            keys.update(r.keys())
    for k in sorted(keys):
        if not any(h in k.lower() for h in LABEL_HINTS):
            continue
        vals = []
        for r in records:
            v = r.get(k) if isinstance(r, dict) else None
            # span lists (e.g. RAGTruth): hallucinated if non-empty
            vals.append(("non-empty list" if v else "empty list") if isinstance(v, list) else str(v))
        c = Counter(vals)
        if len(c) <= 12:
            print("   label-like field '%s': %s" % (k, dict(c)))


def inspect_file(path: Path):
    rel = path.relative_to(DATA)
    print("\n -- %s" % rel)
    recs = load(path)
    if not recs:
        return
    print("   records: %d" % len(recs))
    first = recs[0]
    if isinstance(first, dict):
        print("   fields : %s" % list(first.keys()))
        for i, r in enumerate(recs[:3]):
            print("   sample %d:" % (i + 1))
            for k, v in r.items():
                print("      %-22s %s" % (k, short(v)))
        label_balance(recs)
    else:
        print("   sample :", short(first))


def main():
    if not DATA.exists():
        sys.exit("datasets/ folder not found. Create it and download the data first.")
    for sub in sorted(p for p in DATA.iterdir() if p.is_dir()):
        print("\n" + "=" * 70)
        print("DATASET:", sub.name)
        print("=" * 70)
        files = [
            f
            for f in sorted(sub.rglob("*"))
            if f.is_file()
            and f.suffix in EXTS
            and not (set(f.parts) & SKIP_DIRS)
            and f.stat().st_size > 0
        ]
        if not files:
            print(" (no csv/json/jsonl files found - check the download)")
            continue
        print(" %d data files found, showing up to %d" % (len(files), MAX_FILES))
        for f in files[:MAX_FILES]:
            inspect_file(f)
    print("\nDone. Compare the fields above with Section 4 of the plan and note any differences.")


if __name__ == "__main__":
    main()
