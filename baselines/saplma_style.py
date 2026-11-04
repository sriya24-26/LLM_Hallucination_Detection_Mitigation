"""SAPLMA-style baseline: Path B alone, no clipping and no gate. risk = 1 - p_know_noclip.  Owner: Kriti."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from eval.tune import Method, best_single  # noqa: E402


class SaplmaStyle(Method):
    name = "saplma_style"
    needs = ["p_know_noclip"]

    def prep(self, df):
        df = df.copy()
        df["s_B"] = 1 - df["p_know_noclip"]
        return df

    def risk(self, df):
        return df["s_B"].values

    def tune(self, v, k):
        return best_single(v["s_B"].values, v["label"].values.astype(bool), k)

    def predict(self, df, p):
        return df["s_B"].values > p["tau"]


METHOD = SaplmaStyle()
