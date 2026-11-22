"""Hyperparameter table for the report, built from config.yaml (Report 2 plan, Section 5).
  python scripts/hyperparam_table.py
Prints the table and writes results/hyperparameter_table.csv. Values come from config.yaml; the probe's chosen layer,
clipping and classifier come from results/probe_choice_<model>.json when that file exists. Rows marked 'in code' are
fixed in the scripts, not in config.yaml; check them if you change the code.
"""
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common import CFG, RESULTS  # noqa: E402

P, A, G, T, S, L, E, M = (CFG[k] for k in ("probe", "path_a", "gate", "thresholds", "selfcheck", "llm", "evaluation", "models"))
choice_file = RESULTS / ("probe_choice_%s.json" % M["probe"].split("/")[-1].lower())
choice = json.load(open(choice_file)) if choice_file.exists() else {}
grid = P["classifier_grid"]
rows = [
    ("General", "random seed", CFG["project"]["seed"], "fixed"),
    ("General", "validation / test split", "%d / %d (by example_id)" % (round(100 * CFG["project"]["val_fraction"]), round(100 * (1 - CFG["project"]["val_fraction"]))), "fixed"),
    ("General", "bootstrap resamples / unit", "%s / %s" % (E["bootstrap_resamples"], E["bootstrap_unit"]), "fixed"),
    ("Models", "NLI model (Path A, gate)", M["nli"], "fixed"),
    ("Models", "probe model (Path B)", M["probe"], "fixed (1.5B tried as a comparison)"),
    ("Models", "sentence embedding (gate)", M["embedding"], "fixed"),
    ("Path A", "window / stride / max length (tokens)", "%s / %s / %s" % (A["window_tokens"], A["stride_tokens"], A["max_length"]), "fixed in advance"),
    ("Path A", "max windows per evidence", A["max_windows"], "fixed in advance"),
    ("Gate", "NLI hypothesis template", G["hypothesis_template"], "fixed in advance"),
    ("Gate", "logistic regression C", G["logreg_C"], "fixed in advance"),
    ("Gate", "max NLI windows / embedding max tokens", "%s / %s" % (G["max_windows"], G["embedding_max_tokens"]), "fixed in advance"),
    ("Probe", "layer L", choice.get("layer", P["layer"]), "best mean leave-one-topic-out AUROC over all layers"),
    ("Probe", "clipping percentile (options)", P["clip_percentiles"], "leave-one-topic-out AUROC"),
    ("Probe", "clipping percentile (chosen)", choice.get("clip", "see probe_choice json"), "leave-one-topic-out AUROC"),
    ("Probe", "classifier (chosen) and C", "%s, C=%s" % (choice.get("kind", "?"), choice.get("C", "?")), "leave-one-topic-out AUROC"),
    ("Probe", "MLP hidden units / learning rate / L2", "%s / %s / %s" % (grid["mlp_hidden"], grid["mlp_lr"], grid["mlp_l2"]), "small grid"),
    ("Probe", "logistic regression C (options)", grid["logreg_C"], "small grid"),
    ("Probe", "MLP max epochs / early stopping (in code)", "200 / on (10% held out, scikit-learn defaults)", "fixed"),
    ("Probe", "logistic regression max iterations (in code)", 1000, "fixed"),
    ("Probe", "batch size / max tokens (in code)", "%s / 256" % P["batch_size"], "fixed"),
    ("Thresholds", "grid points per threshold", T["grid_points"], "quantiles of the validation scores"),
    ("Thresholds", "tuning metric / tuned per", "%s / %s" % (T["tuning_metric"], T["tuned_per"]), "grid search on validation, then frozen"),
    ("SelfCheckGPT", "samples k / temperature", "%s / %s" % (S["k_samples"], S["temperature"]), "fixed"),
    ("SelfCheckGPT", "sampler model / max evidence tokens", "%s / %s" % (L["sampler_model"], S["max_evidence_tokens"]), "fixed"),
    ("LLM calls", "rewrite and mitigation model", L["rewrite_model"], "fixed"),
    ("LLM calls", "temperature (rewrite / mitigation) / reasoning effort", "%s / %s / %s" % (L["temperature_rewrite"], L["temperature_mitigation"], L["reasoning_effort"]), "fixed"),
]
df = pd.DataFrame(rows, columns=["component", "hyperparameter", "value", "how it is set"])
df.to_csv(RESULTS / "hyperparameter_table.csv", index=False)
pd.set_option("display.width", 250, "display.max_colwidth", 80)
print(df.to_string(index=False))
print("\nsaved", RESULTS / "hyperparameter_table.csv")
