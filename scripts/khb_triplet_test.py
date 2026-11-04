"""KnowHalBench triplet-fragment check (Sriya, Nov 5). No API calls, validation split only.
  python scripts/khb_triplet_test.py
Scores up to 80 KnowHalBench validation claims per setting with Path A and prints AUROC of (1 - a_entail).
Rough guide: 0.70 or more per setting -> keep the triplets as they are. Between 0.60 and 0.70 -> borderline,
keep them and say so in the report. Below 0.60 -> the plan's fallback is rewriting them as sentences with
Groq (about 900 calls on the 120b model); ask the team before changing claims.parquet, because that changes
the frozen file and every Path A / SelfCheckGPT score for KnowHalBench.
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common import SEED, load_claims  # noqa: E402
from eval.metrics import auroc  # noqa: E402
from models.nli import NLI  # noqa: E402

c = load_claims()
c = c[(c.dataset == "khb") & (c.split == "validation")]
c = c.groupby("setting", group_keys=False).apply(lambda g: g.sample(min(len(g), 80), random_state=SEED))
nli = NLI()
pairs = sum(len(nli.windows(e)) for e in c.evidence)
print("%d claims, %d NLI pairs (about %.0f min at 1.2 s/pair)" % (len(c), pairs, pairs * 1.2 / 60))
t0 = time.perf_counter()
parts = []
for s in range(0, len(c), 30):
    ch = c.iloc[s:s + 30]
    parts.append(nli.path_a(ch.evidence.tolist(), ch.claim_text.tolist())[0])
    done, el = min(s + 30, len(c)), time.perf_counter() - t0
    print("  %d / %d claims, %.1f min elapsed, about %.1f min left" % (done, len(c), el / 60, el / done * (len(c) - done) / 60), flush=True)
ae = np.concatenate(parts)
c = c.assign(s_A=1 - ae)
print("done in %.1f min\n" % ((time.perf_counter() - t0) / 60))
for st, g in c.groupby("setting"):
    print("%-9s n=%3d  positives=%3d  AUROC(1 - a_entail) = %.3f" % (st, len(g), g.label.sum(), auroc(g.label.values, g.s_A.values)))
print("\nlowest-scoring hallucinated and best-supported correct claims, to read by eye:")
for lab in (1, 0):
    g = c[c.label == lab].sort_values("s_A", ascending=(lab == 0)).head(3)
    for _, r in g.iterrows():
        print("  label %d | s_A %.2f | %s" % (lab, r.s_A, r.claim_text))
