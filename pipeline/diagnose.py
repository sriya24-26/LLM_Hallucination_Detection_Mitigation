"""Diagnosis: reliability-weighted risk and the flag rules from Report 2 (Section 3).  Owner: Erina (with Kriti).
  trusted      = g >= tau_g
  Flag A       = trusted AND s_A > tau_A                      -> mitigation: regenerate with the reference
  Flag B       = NOT trusted AND s_B > tau_B                  -> mitigation: hedge or abstain
  Disagreement = trusted AND s_A <= tau_A AND s_B > tau_B     -> mitigation: warn that the source may be unreliable
  predicted positive = any of the three flags
  r = g * s_A + (1 - g) * s_B        continuous score used for AUROC
  d = g * a_entail * s_B             continuous disagreement score (evidence supports the claim, the probe does not)
s_A = 1 - a_entail (Path A risk), s_B = 1 - p_know (Path B risk).
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from eval.tune import Method, best_triple  # noqa: E402


def diagnose(df, tau_g, tau_A, tau_B, use_disagreement=True):
    """Returns (predicted positive, disagreement flag, flag type per claim: 'A', 'B', 'D' or '')."""
    trusted = df["g"].values >= tau_g
    sA, sB = df["s_A"].values, df["s_B"].values
    flag_a = trusted & (sA > tau_A)
    flag_b = ~trusted & (sB > tau_B)
    flag_d = (trusted & (sA <= tau_A) & (sB > tau_B)) if use_disagreement else np.zeros(len(df), bool)
    kind = np.where(flag_a, "A", np.where(flag_b, "B", np.where(flag_d, "D", "")))
    return flag_a | flag_b | flag_d, flag_d, kind


class Ours(Method):
    """g_mode=1 / 0 fixes the gate to always-A / always-B (ablations); pcol='p_know_noclip' is the no-clipping ablation."""

    def __init__(self, name="ours", g_mode=None, use_disagreement=True, pcol="p_know"):
        self.name, self.g_mode, self.use_d, self.pcol = name, g_mode, use_disagreement, pcol
        self.needs = ["a_entail", pcol] + ([] if g_mode is not None else ["g"])

    def prep(self, df):
        df = df.copy()
        if self.g_mode is not None:
            df["g"] = float(self.g_mode)
        df["s_A"] = 1 - df["a_entail"]
        df["s_B"] = 1 - df[self.pcol]
        df["r"] = df["g"] * df["s_A"] + (1 - df["g"]) * df["s_B"]
        df["d"] = df["g"] * df["a_entail"] * df["s_B"]
        return df

    def risk(self, df):
        return df["r"].values

    def tune(self, v, k):
        fixed = None if self.g_mode is None else (0.0 if self.g_mode == 1 else 2.0)  # 2.0 = nothing is ever trusted
        return best_triple(v["g"].values, v["s_A"].values, v["s_B"].values, v["label"].values.astype(bool), k,
                           self.use_d, fixed)

    def predict(self, df, p):
        return diagnose(df, p["tau_g"], p["tau_A"], p["tau_B"], self.use_d)[0]
