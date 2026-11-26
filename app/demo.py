"""Gradio demo of the full pipeline (Report 2 plan, Phase 7).  Owner: Sriya.
  python app/demo.py              local page at http://127.0.0.1:7860
  python app/demo.py --port 7861  another port
Paste a question, a reference passage and an answer. The answer is split into sentences; each sentence gets
  the reliability gate score g (query + passage), Path A risk s_A = 1 - max entailment (roberta-large-mnli),
  Path B risk s_B = 1 - p_know (Qwen2.5-0.5B hidden state + probe),
then the flag rules of Report 2 with the FROZEN thresholds of results/thresholds.json decide the mitigation:
  A regenerate with the reference (one Groq call, optional) | B hedge | D warn that the source may be unreliable.
Needs: cache/gate.joblib, cache/probe_<model>.joblib, results/thresholds.json, the three models in the Hugging Face
cache, and (only for the Groq checkbox) GROQ_API_KEY in .env. CPU only; about 10 to 20 seconds per click.
Do not use --share: it would let strangers spend your Groq quota.
"""
import argparse
import json
import sys
import threading
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from common import CACHE, CFG, RESULTS  # noqa: E402
from pipeline.diagnose import Ours, diagnose  # noqa: E402
from pipeline.mitigate import hedge, regenerate, warn  # noqa: E402

MIN_WORDS = CFG["sampling"]["ragtruth_min_sentence_words"]

COLUMNS = ["sentence", "gate g", "Path A risk", "Path B risk", "flag", "action"]
FLAG_NAMES = {"A": "A: not supported by the reference", "B": "B: unreliable reference, low belief",
              "D": "D: reference supports it, probe disagrees", "": "none"}


class Pipeline:
    """Loads the models once and scores sentences. analyse() only needs .score() and .thr."""

    def load(self):
        import __main__
        from models import probe as probe_mod
        from models.gate import load_embedder, load_gate
        from models.nli import NLI

        __main__.Probe = probe_mod.Probe  # the saved probe was pickled from a script that ran as __main__
        saved = joblib.load(CACHE / ("probe_%s.joblib" % probe_mod.model_tag(CFG["models"]["probe"])))
        self.layer, self.probe = saved["layer"], saved["probe"]
        self.nli, self.emb, self.gate = NLI(), load_embedder(), load_gate()
        self.extractor = probe_mod.Extractor(CFG["models"]["probe"])
        self.thr = json.load(open(RESULTS / "thresholds.json"))["ours"]

    def score(self, query, evidence, claims):
        from models.gate import gate_features

        t, t0 = {}, time.perf_counter()
        g = float(self.gate.predict_proba(gate_features(self.nli, self.emb, [query], [evidence]))[0, 1])
        t["gate"], t0 = time.perf_counter() - t0, time.perf_counter()
        a_entail, _, _ = self.nli.path_a([evidence] * len(claims), claims)
        t["path_a"], t0 = time.perf_counter() - t0, time.perf_counter()
        p_know = self.probe.p_true(np.asarray(self.extractor(claims, batch_size=16)[:, self.layer, :]))
        t["path_b"] = time.perf_counter() - t0
        return g, a_entail, p_know, t


_GROQ_LOCK = threading.Lock()


def regenerate_threadsafe(query, evidence):
    """Gradio runs every click in a worker thread, but llm.py keeps one SQLite connection that may only be used in the
    thread that created it. Open a fresh connection for each call and close it afterwards (llm.py stays unchanged)."""
    import llm

    with _GROQ_LOCK:
        llm._db = None
        try:
            return regenerate(query, evidence)
        finally:
            if getattr(llm, "_db", None) is not None:
                llm._db.close()
                llm._db = None


def split_claims(answer):
    import pysbd

    return [s.strip() for s in pysbd.Segmenter(language="en", clean=False).segment(answer) if len(s.split()) >= MIN_WORDS]


def analyse(pipe, query, evidence, answer, thr_name, use_groq):
    query, evidence, answer = ((x or "").strip() for x in (query, evidence, answer))
    if not (query and evidence and answer):
        return pd.DataFrame(columns=COLUMNS), "", "Please fill in the question, the reference passage and the answer."
    claims = split_claims(answer)
    if not claims:
        return pd.DataFrame(columns=COLUMNS), "", "No sentence with at least %d words was found in the answer." % MIN_WORDS
    g, a_entail, p_know, t = pipe.score(query, evidence, claims)
    d = Ours().prep(pd.DataFrame(dict(claim_text=claims, g=g, a_entail=a_entail, p_know=p_know)))
    p = pipe.thr[thr_name]
    kind = diagnose(d, p["tau_g"], p["tau_A"], p["tau_B"])[2]

    note, regenerated, why = "", None, "the Groq box was not ticked when Check was clicked"
    if use_groq and "A" in kind:
        try:
            regenerated = regenerate_threadsafe(query, evidence) or None
            if regenerated is None:
                why = "Groq returned an empty answer"
        except Exception as e:  # noqa: BLE001 (missing key, daily limit, network)
            why = "Groq call failed: %s" % str(e)[:100]
            note = " %s." % why
    elif use_groq:
        why = "no sentence is flagged A"
    rows, parts, placed = [], [], False
    for claim, k, (_, r) in zip(claims, kind, d.iterrows()):
        if k == "A" and regenerated is not None:
            action = "regenerated from the reference"
            if not placed:
                parts.append(regenerated)
                placed = True
        elif k == "A":
            action = "flagged, not regenerated: %s" % why
            parts.append("%s [flagged: not supported by the reference]" % claim)
        elif k == "B":
            action, _ = "hedged", parts.append(hedge(claim))
        elif k == "D":
            action, _ = "warning added", parts.append(warn(claim))
        else:
            action, _ = "unchanged", parts.append(claim)
        rows.append([claim, round(float(r["g"]), 2), round(float(r["s_A"]), 2), round(float(r["s_B"]), 2), FLAG_NAMES[k], action])
    info = ("Thresholds tuned on **%s**: tau_g = %.3f, tau_A = %.3f, tau_B = %.4f. Latency: gate %.0f ms, Path A %.0f ms, "
            "Path B %.0f ms for %d sentence(s).%s" % (thr_name, p["tau_g"], p["tau_A"], p["tau_B"], 1000 * t["gate"],
                                                      1000 * t["path_a"], 1000 * t["path_b"], len(claims), note))
    return pd.DataFrame(rows, columns=COLUMNS), "\n".join(parts), info


def build_ui(pipe):
    import gradio as gr

    with gr.Blocks(title="Hallucination detection and mitigation") as ui:
        gr.Markdown("# Reliability-weighted hallucination detection and mitigation\n"
                    "Paste a question, a reference passage and an answer to check. Everything runs on the CPU.")
        q = gr.Textbox(label="Question", lines=2, placeholder="Type the question here")
        ev = gr.Textbox(label="Reference passage", lines=6, placeholder="Type the reference passage")
        ans = gr.Textbox(label="Answer to check", lines=5, placeholder="Add the answer to check if right or not")
        with gr.Row():
            thr = gr.Dropdown(choices=list(pipe.thr.keys()), value="halueval", label="Thresholds tuned on")
            groq = gr.Checkbox(value=False, label="Regenerate flagged (A) answers with Groq (needs GROQ_API_KEY in .env)")
        btn = gr.Button("Check")
        table = gr.Dataframe(label="Per-sentence result", headers=COLUMNS)
        final = gr.Textbox(label="Mitigated answer", lines=5)
        info = gr.Markdown()
        btn.click(lambda *a: analyse(pipe, *a), [q, ev, ans, thr, groq], [table, final, info])
    return ui


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=7860)
    a = ap.parse_args()
    pipeline = Pipeline()
    print("Loading the models (about a minute) ...", flush=True)
    pipeline.load()
    print("Ready. Open http://127.0.0.1:%d" % a.port, flush=True)
    build_ui(pipeline).launch(server_port=a.port)