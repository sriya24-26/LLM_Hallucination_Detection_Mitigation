import gc
import json
import os
import platform
import socket
import time
from pathlib import Path

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

import psutil  # noqa: E402
import torch  # noqa: E402
from transformers import AutoModel, AutoModelForSequenceClassification, AutoTokenizer  # noqa: E402
from sentence_transformers import SentenceTransformer  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
NLI = "roberta-large-mnli"
PROBE = "Qwen/Qwen2.5-0.5B-Instruct"
EMB = "sentence-transformers/all-MiniLM-L6-v2"
N_CLAIMS = 48
BATCH = 16

SENTENCES = [
    "Paris is the capital of France.",
    "The chemical symbol for gold is Au.",
    "Mount Everest is located on the border between Nepal and China.",
    "The Pacific Ocean is the largest ocean on Earth.",
    "Albert Einstein developed the theory of general relativity.",
    "The Great Wall of China is visible from the Moon with the naked eye.",
    "Water boils at 100 degrees Celsius at sea level.",
    "The company was founded in 1998 by two university students.",
    "Cairo is the capital city of Egypt and lies on the Nile.",
    "The human heart has four chambers.",
    "Penicillin was discovered by Alexander Fleming in 1928.",
    "Brazil won its first football World Cup in 1950.",
]
claims = (SENTENCES * 4)[:N_CLAIMS]
proc = psutil.Process(os.getpid())
peak_rss = 0.0


def note_ram():
    global peak_rss
    peak_rss = max(peak_rss, proc.memory_info().rss / 1e9)


def per_item_seconds(fn, items):
    fn(items[:2])  # warm-up, not timed
    start = time.perf_counter()
    for i in range(0, len(items), BATCH):
        fn(items[i : i + BATCH])
    return (time.perf_counter() - start) / len(items)


total_ram = psutil.virtual_memory().total / 1e9
print("Computer      :", socket.gethostname(), "|", platform.platform())
print("Processor     :", platform.processor() or "unknown")
print("Cores / RAM   : %s logical cores, %.1f GB" % (os.cpu_count(), total_ram))
print("Torch threads :", torch.get_num_threads())
print("Claims timed  : %d (batch size %d)\n" % (N_CLAIMS, BATCH))

# ---- Path B: hidden states of the final token, all layers --------------------
print("Path B (%s) ..." % PROBE)
tok = AutoTokenizer.from_pretrained(PROBE, padding_side="left")
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
model = AutoModel.from_pretrained(PROBE, torch_dtype=torch.float32).eval()
note_ram()


@torch.inference_mode()
def path_b(batch):
    enc = tok(batch, return_tensors="pt", padding=True)
    out = model(**enc, output_hidden_states=True)
    return torch.stack([h[:, -1, :] for h in out.hidden_states], dim=1)


shape = tuple(path_b(claims[:2]).shape[1:])
path_b_s = per_item_seconds(path_b, claims)
note_ram()
print("  hidden states per claim: %d layers x %d dims" % shape)
print("  %.3f s per claim" % path_b_s)
del model, tok
gc.collect()

# ---- NLI: roberta-large-mnli on 200-token evidence windows ------------------
print("\nNLI (%s) ..." % NLI)
nli_tok = AutoTokenizer.from_pretrained(NLI)
nli = AutoModelForSequenceClassification.from_pretrained(NLI).eval()
note_ram()
ids = nli_tok(" ".join(SENTENCES * 4), add_special_tokens=False)["input_ids"][:200]
premise = nli_tok.decode(ids)


@torch.inference_mode()
def nli_fn(batch):
    enc = nli_tok(
        [premise] * len(batch), batch, return_tensors="pt", padding=True, truncation=True, max_length=512
    )
    return nli(**enc).logits.softmax(-1)


nli_s = per_item_seconds(nli_fn, claims)
note_ram()
print("  %.3f s per (200-token window, claim) pair" % nli_s)
del nli, nli_tok
gc.collect()

# ---- Embedding model for the gate ---------------------------------------------
print("\nEmbedding (%s) ..." % EMB)
emb = SentenceTransformer(EMB, device="cpu")
emb_s = per_item_seconds(lambda b: emb.encode(b), claims)
note_ram()
print("  %.4f s per sentence" % emb_s)

# ---- Summary --------------------------------------------------------------------
ram_limit = 0.7 * total_ram
checks = [
    ("Path B  <= 1.0 s/claim", path_b_s <= 1.0),
    ("NLI     <= 2.0 s/pair", nli_s <= 2.0),
    ("Peak RAM <= 70%% of %.1f GB" % total_ram, peak_rss <= ram_limit),
]
print("\n== Result (planning thresholds from the plan) ==")
for label, ok in checks:
    print("  %-34s %s" % (label, "OK" if ok else "CHECK"))
print("  Peak RAM used: %.1f GB" % peak_rss)

row = "| %s | %s | %s | %.1f | %.2f | %.2f |" % (
    socket.gethostname(),
    platform.processor() or "unknown",
    os.cpu_count(),
    total_ram,
    path_b_s,
    nli_s,
)
print("\nTracker row (CPU | cores | RAM GB | Path B s/claim | NLI s/pair):")
print(row)

out_dir = ROOT / "cache"
out_dir.mkdir(exist_ok=True)
out_file = out_dir / ("timing_%s.json" % socket.gethostname())
out_file.write_text(
    json.dumps(
        {
            "computer": socket.gethostname(),
            "processor": platform.processor(),
            "cores": os.cpu_count(),
            "ram_gb": round(total_ram, 1),
            "torch_threads": torch.get_num_threads(),
            "path_b_s_per_claim": round(path_b_s, 4),
            "nli_s_per_pair": round(nli_s, 4),
            "embedding_s_per_sentence": round(emb_s, 5),
            "peak_ram_gb": round(peak_rss, 2),
        },
        indent=2,
    )
)
print("\nSaved:", out_file)