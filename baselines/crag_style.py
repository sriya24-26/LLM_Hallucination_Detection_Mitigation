"""CRAG-style baseline: gate, then Path A when the evidence is trusted; when untrusted predict 'not flagged'
(there is no web search to fall back on). Flag = (g >= tau_g) AND (s_A > tau_A). AUROC score = g * s_A.  Owner: Kriti."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from eval.tune import Method, best_pair  # noqa: E402


class CragStyle(Method):
    name = "crag_style"
    needs = ["a_entail", "g"]

    def prep(self, df):
        df = df.copy()
        df["s_A"] = 1 - df["a_entail"]
        df["crag"] = df["g"] * df["s_A"]
        return df

    def risk(self, df):
        return df["crag"].values

    def tune(self, v, k):
        return best_pair(v["g"].values, v["s_A"].values, v["label"].values.astype(bool), k)

    def predict(self, df, p):
        return (df["g"].values >= p["tau_g"]) & (df["s_A"].values > p["tau_A"])


METHOD = CragStyle()
