"""Phase 1 gate: validate cache/claims.parquet, print counts, write results/claims_manifest.json.
  python scripts/freeze_claims.py                      check + write manifest (data owner)
  python scripts/freeze_claims.py --verify             teammates: does my claims.parquet match the manifest?
  python scripts/freeze_claims.py --spot 10 --datasets halueval controlled --seed 1    human spot check
"""
import argparse
import hashlib
import json
import sys
from datetime import date
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parent.parent
CFG = yaml.safe_load(open(ROOT / "config.yaml", encoding="utf-8"))
PATH = ROOT / CFG["project"]["paths"]["cache"] / "claims.parquet"
MANIFEST = ROOT / CFG["project"]["paths"]["results"] / "claims_manifest.json"
EXPECTED = {"halueval": {"qa"}, "controlled": {"unreliable", "reliable", "reliable_wrongclaim"},
            "khb": {"zero", "noisy", "accurate"}, "ragtruth": {"summary", "qa", "data2txt"}}

ap = argparse.ArgumentParser()
ap.add_argument("--verify", action="store_true")
ap.add_argument("--spot", type=int, default=0)
ap.add_argument("--datasets", nargs="*")
ap.add_argument("--seed", type=int, default=0)
args = ap.parse_args()


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


df = pd.read_parquet(PATH)

if args.verify:
    ok = json.loads(MANIFEST.read_text())["sha256"] == sha256(PATH)
    print("MATCH: same claims.parquet as the manifest" if ok else "MISMATCH: download claims.parquet from Drive again")
    sys.exit(0 if ok else 1)

if args.spot:
    for name in args.datasets or sorted(df.dataset.unique()):
        print("\n=== %s ===" % name)
        for _, r in df[df.dataset == name].sample(args.spot, random_state=args.seed).iterrows():
            print("[%s | label %d] query: %s\n  claim: %s\n  evidence: %s...\n"
                  % (r["setting"], r["label"], r["query"][:120], r["claim_text"], r["evidence"][:200].replace("\n", " ")))
    sys.exit(0)

errors, warnings = [], []
if not df.claim_id.is_unique:
    errors.append("duplicate claim_id")
for col in ["claim_id", "claim_text", "evidence", "query", "example_id"]:
    n = int((df[col].fillna("").astype(str).str.strip() == "").sum())
    if n:
        errors.append("%d empty %s" % (n, col))
if not set(df.label.unique()) <= {0, 1}:
    errors.append("label is not 0/1")
if not set(df.split.unique()) <= {"validation", "test"}:
    errors.append("bad split values")
leak = int((df.groupby("example_id").split.nunique() > 1).sum())
if leak:
    errors.append("%d examples appear in both splits" % leak)
for ds, settings in EXPECTED.items():
    have = set(df[df.dataset == ds].setting.unique())
    if not have:
        errors.append("dataset missing: " + ds)
    elif settings - have:
        warnings.append("%s: missing settings %s" % (ds, sorted(settings - have)))
for (ds, sp), g in df[df.dataset != "controlled"].groupby(["dataset", "split"]):
    pos = int(g.label.sum())
    if min(pos, len(g) - pos) < 20:
        warnings.append("%s %s: fewer than 20 claims in one class (%d positives of %d)" % (ds, sp, pos, len(g)))

print(df.groupby(["dataset", "setting", "split", "label"]).size().unstack(fill_value=0).to_string())
print("\nclaims per dataset:", df.dataset.value_counts().to_dict())
for w in warnings:
    print("WARNING:", w)
for e in errors:
    print("ERROR:", e)
if errors:
    sys.exit(1)
MANIFEST.parent.mkdir(parents=True, exist_ok=True)
MANIFEST.write_text(json.dumps({"date": date.today().isoformat(), "file": PATH.name, "rows": len(df),
                                "sha256": sha256(PATH), "per_dataset": df.dataset.value_counts().to_dict()}, indent=2))
print("\nOK. Manifest written to", MANIFEST)