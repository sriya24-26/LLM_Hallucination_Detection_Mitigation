"""KnowHalBench triplet-fragment check (Sriya, Nov 5). No API calls, validation split only.
  python scripts/khb_triplet_test.py
Scores up to 80 KnowHalBench validation claims per setting with Path A and prints AUROC of (1 - a_entail).
Rough guide: 0.70 or more per setting -> keep the triplets as they are. Between 0.60 and 0.70 -> borderline,
keep them and say so in the report. Below 0.60 -> the plan's fallback is rewriting them as sentences with
Groq (about 900 calls on the 120b model); ask the team before changing claims.parquet, because that changes
the frozen file and every Path A / SelfCheckGPT score for KnowHalBench.
Accurate has only about 10 validation positives, so judge mainly on noisy and zero.
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common import SEED, load_claims  # noqa: E402
from eval.metrics import auroc  # noqa: E402
from models.nli import NLI  # noqa: E402

STEP = 30  # claims per progress line

c = load_claims()
c = c[(c.dataset == "khb") & (c.split == "validation")]
c = pd.concat([g.sample(min(len(g), 80), random_state=SEED) for _, g in c.groupby("setting")], ignore_index=True)
nli = NLI()
pairs = sum(len(nli.windows(e)) for e in c.evidence)
print("%d claims, %d NLI pairs (about %.0f min at 1.2 s/pair)" % (len(c), pairs, pairs * 1.2 / 60), flush=True)

t0 = time.perf_counter()
parts = []
for s in range(0, len(c), STEP):
    ch = c.iloc[s:s + STEP]
    parts.append(nli.path_a(ch.evidence.tolist(), ch.claim_text.tolist())[0])
    done, el = min(s + STEP, len(c)), time.perf_counter() - t0
    print("  %d / %d claims, %.1f min elapsed, about %.1f min left"
          % (done, len(c), el / 60, el / done * (len(c) - done) / 60), flush=True)
c = c.assign(s_A=1 - np.concatenate(parts))
print("\ndone in %.1f min\n" % ((time.perf_counter() - t0) / 60))

for st, g in c.groupby("setting"):
    print("%-9s n=%3d  positives=%3d  AUROC(1 - a_entail) = %.3f" % (st, len(g), g.label.sum(), auroc(g.label.values, g.s_A.values)))

print("\nto read by eye: hallucinated claims that Path A thinks are supported (lowest s_A), then correct claims")
print("that Path A cannot verify (highest s_A). Many of the second kind means the triplet wording hurts NLI.")
for lab, asc in ((1, True), (0, False)):
    for _, r in c[c.label == lab].sort_values("s_A", ascending=asc).head(4).iterrows():
        print("  label %d | s_A %.2f | %s" % (lab, r.s_A, r.claim_text))