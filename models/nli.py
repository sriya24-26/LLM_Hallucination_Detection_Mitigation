"""Path A and shared NLI utilities: roberta-large-mnli, CPU, float32.  Owner: Erina.
Smoke test:  python models/nli.py
"""
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common import CFG  # noqa: E402


def slice_windows(ids, window, stride, max_windows):
    """Overlapping token windows [start, start+window); stops once the end is covered; keeps the first max_windows."""
    if len(ids) <= window:
        return [list(ids)]
    out = []
    for start in range(0, len(ids), stride):
        out.append(list(ids[start:start + window]))
        if start + window >= len(ids):
            break
    return out[:max_windows]


class NLI:
    def __init__(self, name=None):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.torch = torch
        name = name or CFG["models"]["nli"]
        self.tok = AutoTokenizer.from_pretrained(name)
        self.tok.model_max_length = 10 ** 6  # only silences the long-text warning; pairs are truncated explicitly below
        self.model = AutoModelForSequenceClassification.from_pretrained(name, torch_dtype=torch.float32).eval()
        lab = {v.lower(): k for k, v in self.model.config.id2label.items()}
        self.order = [lab["contradiction"], lab["neutral"], lab["entailment"]]
        self._cache = {}

    def windows(self, text, max_windows=None):
        """Evidence split into ~200-token windows (stride 100). Short text is returned unchanged."""
        pa = CFG["path_a"]
        max_windows = max_windows or pa["max_windows"]
        key = (text, max_windows)
        if key not in self._cache:
            ids = self.tok(text, add_special_tokens=False)["input_ids"]
            if len(ids) <= pa["window_tokens"]:
                self._cache[key] = [text]
            else:
                parts = slice_windows(ids, pa["window_tokens"], pa["stride_tokens"], max_windows)
                self._cache[key] = [self.tok.decode(p) for p in parts]
        return self._cache[key]

    def count_pairs(self, evidences, max_windows=None):
        return sum(len(self.windows(e, max_windows)) for e in evidences)

    def probs(self, premises, hypotheses, batch_size=16):
        """(n, 3) array with columns contradiction, neutral, entailment."""
        n = len(premises)
        out = np.zeros((n, 3), dtype=np.float32)
        order = sorted(range(n), key=lambda i: len(premises[i]) + len(hypotheses[i]))
        with self.torch.inference_mode():
            for b in range(0, n, batch_size):
                idx = order[b:b + batch_size]
                enc = self.tok([premises[i] for i in idx], [hypotheses[i] for i in idx], return_tensors="pt",
                               padding=True, truncation=True, max_length=CFG["path_a"]["max_length"])
                p = self.model(**enc).logits.softmax(-1).numpy()
                out[idx] = p[:, self.order]
        return out

    def path_a(self, evidences, claims, batch_size=16, max_windows=None):
        """Per claim: a_entail = max entailment over evidence windows, a_contra = max contradiction over windows.
        Path A risk is s_A = 1 - a_entail."""
        prem, hyp, owner = [], [], []
        for k, (e, c) in enumerate(zip(evidences, claims)):
            for w in self.windows(e, max_windows):
                prem.append(w)
                hyp.append(c)
                owner.append(k)
        p = self.probs(prem, hyp, batch_size)
        owner = np.array(owner)
        n = len(claims)
        a_entail = np.zeros(n, np.float32)
        a_contra = np.zeros(n, np.float32)
        np.maximum.at(a_entail, owner, p[:, 2])
        np.maximum.at(a_contra, owner, p[:, 0])
        return a_entail, a_contra, np.bincount(owner, minlength=n)


if __name__ == "__main__":
    nli = NLI()
    prem = ["Paris is the capital of France."] * 3
    hyp = ["Paris is the capital city of France.", "Paris is the capital of Germany.", "France has a large population."]
    t = time.perf_counter()
    p = nli.probs(prem, hyp)
    dt = time.perf_counter() - t
    print("columns: contradiction  neutral  entailment")
    for h, row in zip(hyp, p):
        print("  %.2f  %.2f  %.2f | %s" % (row[0], row[1], row[2], h))
    print("expected: entailment high / contradiction high / neutral high.  %.2f s for 3 pairs" % dt)
