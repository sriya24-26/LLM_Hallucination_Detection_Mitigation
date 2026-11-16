# RESULTS.md: evaluation run log

Owner: Kriti (baselines and evaluation). Laptop: DESKTOP-1JVUD0O (Intel 4 cores, 17 GB RAM, CPU only).
Code at commit `b9a84c2` (results/ added in the commit that contains this file).

## Inputs

| File | From | SHA256 |
| --- | --- | --- |
| `cache/claims.parquet` | Sriya (frozen) | matches `results/claims_manifest.json` (`freeze_claims.py --verify` = MATCH) |
| `cache/gate.joblib` | Erina | `20E2C4FC6296117C9F6C3182515E162DACC38C2D40138B2D1C41463A7453FA83` (matches Erina's) |
| `config.yaml` | repo | `4AD5FB2B68FE7E22D913A9CDE5A09A1280320DA57F82720514AA2369F48E2B42` |

Shard files merged: `scores_a_shard{1,2,3}of3`, `scores_gate_shard{1,2,3}of3`, `scores_selfcheck_shard{1,2,3}of3` (Erina 1/3, Sriya 2/3, Kriti 3/3, weights 55,25,20) and `scores_pathb.parquet` (Michelle, probe layer 15). SelfCheck samples: all four `selfcheck_samples_Nof4.jsonl`, 792 prompts x 4 samples.

## Commands run (project root, venv active)

| Date | Command | Result |
| --- | --- | --- |
| Nov 8 | `python pipeline/score_all.py --part a --shard 3/3 --weights 55,25,20` | 788 claims, 2,888 NLI pairs |
| Nov 13 | `python pipeline/score_all.py --part gate --shard 3/3 --weights 55,25,20` | 182 unique pairs |
| Nov 13 | `python baselines/selfcheck.py score --shard 3/3 --weights 55,25,20` | 718 claims, 0 without samples |
| Nov 13 | `python pipeline/score_all.py --merge` | 3,941 rows; missing only selfcheck = 351 (controlled) |
| Nov 13 | `python eval/tune.py` | validation split only; FROZEN `results/thresholds.json` |
| Nov 13 | `python eval/tables.py` | test split; 5 CSVs in `results/`; rerun after adding Sriya's latency files (thresholds unchanged) |
| Nov 15 | `python scripts/sanity_scores.py` | validation only; see note below |

Note: `sanity_scores.py` was run after tuning (it was added to the repo after tuning had been done). It only reads data and changed nothing.

## Outputs

| File | SHA256 |
| --- | --- |
| `cache/scores.parquet` (on Drive, not in git) | `321B44D31456901827B75CC56E840290F62D1CD11468A22D310E41CCD73A0E43` |
| `results/thresholds.json` | `E539D9B26FB774DD9C14C02CFD1E131FE9C0987FF14AE84CCB3D0F669963C4F4` |
| `results/main_table.csv` | `11F7B9BDB90AA186D65F1EE4B06D2DA0D67F83FA0107880B26AE73281C95A305` |

Cross-check (Sriya): `eval/tables.py` with this `scores.parquet` and `thresholds.json` must give the same `main_table.csv` hash.

## Sanity check notes (validation AUROC)

- Path A: 0.70 to 0.90 on every dataset and setting (KnowHalBench Zero 0.904). SelfCheck above 0.5 everywhere.
- Path B (probe): CHECK lines, all near 0.5 (0.46 to 0.49), plus RAGTruth QA 0.128 on 15 positives (0.535 on test). Not a sign or merge error: p_know spans 0 to 1 and some subsets are above 0.5 (KHB Accurate 0.649). Leave-one-topic-out AUROC on the true/false data is 0.777, so the probe does not transfer from true/false statements to the evaluation claims.
- Latency rows for the report: DESKTOP-S2NH3PN (Sriya, reference laptop).
