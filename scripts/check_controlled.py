"""Which controlled 'unreliable' pairs are aligned, i.e. the claim states the phrase inserted into the passage?
Writes results/controlled_aligned_ids.txt (example_ids). Reads claims.parquet only, changes nothing.
Run from the project root:  python scripts/check_controlled.py
"""
import difflib
import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
d = pd.read_parquet(ROOT / "cache" / "claims.parquet")
d = d[d.dataset == "controlled"]
u = d[d.setting == "unreliable"].set_index("example_id")
r = d[d.setting == "reliable"].set_index("example_id")


def words(s):
    return re.findall(r"\w+", s.lower())


def inserted(orig, edited):
    a, b = words(orig), words(edited)
    ops = difflib.SequenceMatcher(None, a, b, autojunk=False).get_opcodes()
    return [" ".join(b[j1:j2]) for tag, i1, i2, j1, j2 in ops if tag in ("replace", "insert")]


aligned, rows = [], []
for ex in u.index:
    ins = inserted(r.loc[ex, "evidence"], u.loc[ex, "evidence"])
    claim = " ".join(words(u.loc[ex, "claim_text"]))
    if any(len(p) > 2 and p in claim for p in ins):
        aligned.append(ex)
    else:
        rows.append((ins[:2], u.loc[ex, "claim_text"]))
print("unreliable pairs: %d | aligned: %d | not aligned: %d" % (len(u), len(aligned), len(rows)))
for ins, claim in rows[:6]:
    print("\nINSERTED:", ins, "\nCLAIM   :", claim)
out = ROOT / "results" / "controlled_aligned_ids.txt"
out.parent.mkdir(exist_ok=True)
out.write_text("\n".join(aligned))
print("\nsaved %d ids -> %s" % (len(aligned), out))