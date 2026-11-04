"""RefChecker-style baseline: Path A alone (always trust the reference). risk = 1 - max entailment.  Owner: Kriti."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from eval.tune import Method, best_single  # noqa: E402


class RefCheckerStyle(Method):
    name = "refchecker_style"
    needs = ["a_entail"]

    def prep(self, df):
        df = df.copy()
        df["s_A"] = 1 - df["a_entail"]
        return df

    def risk(self, df):
        return df["s_A"].values

    def tune(self, v, k):
        return best_single(v["s_A"].values, v["label"].values.astype(bool), k)

    def predict(self, df, p):
        return df["s_A"].values > p["tau"]


METHOD = RefCheckerStyle()
