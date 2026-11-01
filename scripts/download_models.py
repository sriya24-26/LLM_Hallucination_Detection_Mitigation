"""Phase 0, step 3: download the three models (about 3 GB) once per laptop.

Run from the project root:  python scripts/download_models.py
Safe to re-run: files already in the Hugging Face cache are not downloaded again.
"""
import os
import shutil
from pathlib import Path

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")  # harmless warning on Windows

from transformers import AutoModel, AutoModelForSequenceClassification, AutoTokenizer  # noqa: E402
from sentence_transformers import SentenceTransformer  # noqa: E402

NLI = "roberta-large-mnli"
PROBE = "Qwen/Qwen2.5-0.5B-Instruct"
EMB = "sentence-transformers/all-MiniLM-L6-v2"

cache_dir = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface"))
free_gb = shutil.disk_usage(Path.home()).free / 1e9
print("Hugging Face cache :", cache_dir)
print("Free disk space    : %.1f GB (need about 4 GB)" % free_gb)
if free_gb < 4:
    raise SystemExit("Not enough free disk space. Free up space and run again.")

print("\n[1/3] NLI model for the gate and Path A:", NLI)
AutoTokenizer.from_pretrained(NLI)
AutoModelForSequenceClassification.from_pretrained(NLI)

print("\n[2/3] Small model for Path B:", PROBE)
AutoTokenizer.from_pretrained(PROBE)
AutoModel.from_pretrained(PROBE)

print("\n[3/3] Sentence embedding model for the gate:", EMB)
SentenceTransformer(EMB, device="cpu")

print("\nAll three models downloaded.")