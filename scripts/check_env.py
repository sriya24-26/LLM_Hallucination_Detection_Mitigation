"""Quick environment check (Phase 0).   python scripts/check_env.py"""
import hashlib
import json
import sys
from importlib import metadata
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
print("python", sys.version.split()[0], "(need 3.10 to 3.12)")
for pkg in ["torch", "transformers", "sentence-transformers", "scikit-learn", "pandas", "pyarrow", "pysbd", "groq",
            "PyYAML", "tqdm", "psutil", "joblib", "matplotlib"]:
    try:
        print("  %-22s %s" % (pkg, metadata.version(pkg)))
    except metadata.PackageNotFoundError:
        print("  %-22s MISSING%s" % (pkg, " (optional, only for the layer plot)" if pkg == "matplotlib" else ""))
env = ROOT / ".env"
print(".env with GROQ_API_KEY:", env.exists() and "GROQ_API_KEY" in env.read_text())
for f in ["claims.parquet", "truefalse.parquet"]:
    print("cache/%s:" % f, (ROOT / "cache" / f).exists())
man, claims = ROOT / "results" / "claims_manifest.json", ROOT / "cache" / "claims.parquet"
if man.exists() and claims.exists():
    h = hashlib.sha256(claims.read_bytes()).hexdigest()
    print("claims.parquet matches manifest:", h == json.loads(man.read_text())["sha256"])
