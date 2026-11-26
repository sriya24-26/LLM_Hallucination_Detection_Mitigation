# Techniques for Hallucination Detection and Mitigation

Reliability-weighted detection and mitigation of LLM hallucinations, implemented to run on a CPU only.
Course project for CS F425 Deep Learning, BITS Pilani, Dubai Campus.
Authors: Erina Abid, Michelle Moses, Sriya Sreejith, Kriti Wadhawan.

## What it does

An answer is split into sentences (claims). Each claim gets three scores:

| Component | What it asks | Model |
|---|---|---|
| Reliability gate (`g`) | Is the evidence passage relevant to the question? | logistic regression on a zero-shot NLI score and a MiniLM cosine |
| Path A (`s_A = 1 - a_entail`) | Does the passage support the claim? | `roberta-large-mnli` on 200-token windows |
| Path B (`s_B = 1 - p_know`) | Does a small model believe the claim? | probe on the hidden states of `Qwen2.5-0.5B-Instruct` |

The diagnosis step raises a flag, and the flag selects the mitigation:

| Flag | Rule | Mitigation |
|---|---|---|
| A | trusted evidence (`g >= tau_g`) and `s_A > tau_A` | regenerate the answer with the reference (Groq) |
| B | untrusted evidence and `s_B > tau_B` | hedge or abstain |
| D | trusted, `s_A <= tau_A` and `s_B > tau_B` (evidence supports it, probe disagrees) | return the claim with a warning that the source may be unreliable |

The continuous score used for AUROC is `r = g*s_A + (1-g)*s_B`; the disagreement score is `d = g*a_entail*s_B`.
Nothing is fine-tuned: the NLI model, Qwen and MiniLM are used frozen, and only the small probe and the gate are trained.

## Results at a glance (test split, AUROC)

| Dataset | Ours | SelfCheckGPT | SAPLMA-style | RefChecker-style | CRAG-style |
|---|---|---|---|---|---|
| HaluEval | 0.743 | 0.815 | 0.500 | 0.763 | 0.759 |
| KnowHalBench | 0.832 | 0.684 | 0.489 | 0.844 | 0.813 |
| RAGTruth | 0.691 | 0.615 | 0.520 | 0.740 | 0.706 |

The NLI path is the strongest local detector. The probe works on its own training task (leave-one-topic-out AUROC 0.777) but is at chance on claim sentences, so the pipeline does not outperform the reference-only baseline. All tables, with five metrics and bootstrap intervals, are in `results/`.

## Repository layout

```
README.md  requirements.txt  config.yaml  RESULTS.md  common.py  llm.py
app/        demo.py                         Gradio demo
data/       truefalse.py halueval.py khb.py ragtruth.py controlled.py splits.py
models/     nli.py (Path A)  gate.py  probe.py (Path B)
pipeline/   score_all.py  diagnose.py  mitigate.py
baselines/  selfcheck.py  saplma_style.py  refchecker_style.py  crag_style.py
eval/       metrics.py  tune.py  tables.py
scripts/    check_env.py download_models.py inspect_data.py timing.py freeze_claims.py
            check_halueval.py check_controlled.py build_nq_zero.py khb_triplet_test.py
            sanity_scores.py hyperparam_table.py
results/    thresholds.json, result tables (*.csv), probe sweep files, mitigation files, id lists, claims manifest
```

Not in the repository (created locally, see below): `cache/` (claims, scores, models' outputs, API cache), `datasets/` (raw data), `.env` (API key), the virtual environment.

## Setup

1. Python 3.10 to 3.12. Create and activate a fresh virtual environment, then install the pinned packages (the CPU build of PyTorch comes from its own index):
   ```
   python -m venv venv
   venv\Scripts\activate
   pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu
   ```
   The demo also needs `pip install gradio`; install it last, in its own environment if you want the pinned versions unchanged.
2. Groq key (needed only for claim rewriting, SelfCheckGPT sampling, mitigation and the demo's regeneration): create a file `.env` in the project root with `GROQ_API_KEY=your_key`. Never commit it.
3. Models (about 3 GB, downloaded once into the Hugging Face cache): `python scripts/download_models.py`
4. Check the environment: `python scripts/check_env.py`

All values (seed 42, the 30/70 split, model names, window sizes, grids, Groq models and limits) are in `config.yaml`; no script hard-codes them.

## Data

Raw data goes in `datasets/` (git-ignored); `python scripts/inspect_data.py` prints what it finds.

| Folder | Source |
|---|---|
| `datasets/truefalse/` | the six topic CSVs of Azaria and Mitchell (not the generated file) |
| `datasets/refchecker/` | clone of `amazon-science/RefChecker`; KnowHalBench's Noisy and Accurate evidence files come from `benchmark/data/download_data.sh`, and the Zero evidence from `python scripts/build_nq_zero.py` (the original bucket is private, so it is rebuilt from the Hugging Face copy of Natural Questions) |
| `datasets/ragtruth/` | clone of `ParticleMedia/RAGTruth` (`dataset/` inside) |
| `datasets/halueval/` | clone of `RUCAIBox/HaluEval` (`data/qa_data.json`) |

## Running the pipeline, in order

Run everything from the project root. Every slow result is cached, so reruns are cheap.

**1. Build the claims** (do this once; the claim file is then frozen and identified by its hash)
```
python data/truefalse.py        # 6,077 probe-training statements -> cache/truefalse.parquet
python data/halueval.py         # 600 claims (uses Groq, cached)
python data/controlled.py       # controlled unreliable-evidence set, 351 pairs from 117 questions (uses Groq, cached)
python data/ragtruth.py         # 2,090 sentences
python data/khb.py              # 900 claims (Zero, Noisy, Accurate)
python data/splits.py           # 30% validation / 70% test by example, seed 42 -> cache/claims.parquet (3,941 claims)
python scripts/freeze_claims.py # validate, print counts, write results/claims_manifest.json
```
`python scripts/freeze_claims.py --verify` checks a copy of `claims.parquet` against the manifest; `--spot 10 --datasets halueval --seed 1` prints random claims to read.
Do not rebuild the claims on another machine: the Groq rewrites cost calls and are not guaranteed to reproduce the same text. Share the frozen file instead.

**2. Path B probe**
```
python models/probe.py extract --only truefalse   # hidden states of all layers
python models/probe.py sweep                       # leave-one-topic-out: layer, clipping, classifier
python models/probe.py extract --only claims
python models/probe.py final                       # trains the probe, writes cache/scores_pathb.parquet
```
`--model Qwen/Qwen2.5-1.5B-Instruct` runs the larger-model comparison (use `--only truefalse` and `sweep`).

**3. Gate and Path A scores** (sharded so several laptops can share the work; `--weights` splits in proportion to speed)
```
python models/gate.py train                                          # writes cache/gate.joblib (share one file)
python pipeline/score_all.py --part gate --shard 1/3 --weights 55,25,20
python pipeline/score_all.py --part a    --shard 1/3 --weights 55,25,20
```
Run the same commands with shard `2/3` and `3/3` on the other laptops.

**4. SelfCheckGPT baseline** (Groq `gpt-oss-20b`; the free tier limits each account to 200,000 tokens a day, so sampling takes about two days over four accounts)
```
python baselines/selfcheck.py sample --shard 1/4
python baselines/selfcheck.py score  --shard 1/3 --weights 55,25,20   # after all four sample files are in cache/
```

**5. Merge, check, tune, evaluate**
```
python pipeline/score_all.py --merge     # -> cache/scores.parquet (3,941 rows)
python scripts/sanity_scores.py          # validation AUROC of each raw score, before tuning
python eval/tune.py                      # grid search on the validation split, writes results/thresholds.json (refuses to overwrite)
python eval/tables.py                    # test-split tables in results/
```

**6. Mitigation check and extras**
```
python pipeline/mitigate.py              # worked examples of the three flags and correction / harm rates (about 60 Groq calls)
python scripts/hyperparam_table.py       # hyperparameter table from config.yaml
python scripts/timing.py                 # latency of each component on this machine
python app/demo.py                       # Gradio demo at http://127.0.0.1:7860
```

## Reproducing the tables without the models or an API key

Given the frozen `cache/claims.parquet` and the merged `cache/scores.parquet`, the committed `results/thresholds.json` reproduces the result tables with only pandas, numpy, scikit-learn, pyarrow and PyYAML:
```
python scripts/freeze_claims.py --verify
python eval/tables.py
```
The expected hash of `claims.parquet` is in `results/claims_manifest.json`; the run log with file hashes is in `RESULTS.md`.

## Outputs in `results/`

| File | Content |
|---|---|
| `thresholds.json` | frozen thresholds of every method (tuned on validation only) |
| `main_table.csv`, `per_setting_table.csv` | five methods, five metrics, bootstrap intervals, per dataset and per setting |
| `controlled_table.csv` | the disagreement flag on the controlled set |
| `ablation_table.csv`, `latency_table.csv` | ablations and component latency |
| `probe_*` | layer sweep, grid and chosen probe, per topic |
| `mitigation_*` | worked examples, per-claim check, correction and harm rates |
| `halueval_overlap_ids.txt`, `controlled_aligned_ids.txt` | question ids used for the robustness subsets |
| `claims_manifest.json` | hash and counts of the frozen claim file |

## Notes and known limitations

- Thresholds are tuned per dataset on claim types that differ (full sentences, triplet fragments), so one threshold set does not fit any text. The demo lets you choose which set to use.
- The RAGTruth summarization and data-to-text queries are fixed stand-ins, which makes the gate uninformative there.
- Three baselines are CPU-scale re-implementations, not the original systems, and SelfCheckGPT uses a proxy sampler.
- Only the 0.5B probe scored the claims; the 1.5B probe was evaluated on the true-false data.
- Label noise exists in HaluEval (43 of 300 questions have a hallucinated answer containing the right one; results are also reported without them) and KnowHalBench.
