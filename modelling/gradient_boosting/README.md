# Gradient boosting (Person 4): LightGBM and XGBoost

Open **[gbdt_baselines.ipynb](gbdt_baselines.ipynb)**. It is the complete
implementation and the executed report for this slot: everything is defined in
Part A and run in Part B. The write-up is
[`docs/gbdt_report.md`](../../docs/gbdt_report.md).

## Open and run

1. Build the shared splits first (`pipeline/00_download_datasets.py`, then the
   notebooks in `pipeline/notebooks/` in order). The notebook refuses to run if
   any partition's row count differs from the split card.
2. Install the recorded environment from the repository root:

   ```bash
   python -m pip install -r modelling/gradient_boosting/requirements.txt
   ```

3. Run all cells. With `RUN_TRAINING = False` (the default) it only reads the
   committed files in `results/gbdt/` and `figures/gbdt/` and takes seconds.
   Set `RUN_TRAINING = True` to reproduce everything (about five hours,
   serially), and `RUN_OOF = True` to also rebuild the out-of-fold predictions.

For the long runs, [`run_stages.py`](run_stages.py) executes the notebook's own
code stage by stage from a terminal, with live logs, the two libraries in
parallel, and one process per XGBoost model. This is how the committed results
were produced, in about three and a half hours:

```bash
python modelling/gradient_boosting/run_stages.py audit
python modelling/gradient_boosting/run_stages.py tune --families xgb    # GPU, in parallel with:
python modelling/gradient_boosting/run_stages.py tune --families lgbm   # CPU
python modelling/gradient_boosting/run_stages.py fit --families xgb --isolate
python modelling/gradient_boosting/run_stages.py fit --families lgbm
python modelling/gradient_boosting/run_stages.py explain
python modelling/gradient_boosting/run_stages.py oof --families xgb --isolate
python modelling/gradient_boosting/run_stages.py oof --families lgbm
python modelling/gradient_boosting/run_stages.py export
```

Every stage skips work that is already done, so an interrupted run resumes.
Set `GBDT_THREADS` to share the CPU between two parallel runs.

Hardware: XGBoost uses an NVIDIA GPU when one is available and falls back to the
CPU otherwise; LightGBM always runs on the CPU. The completed run used an
RTX 5060 Laptop GPU (8 GB) and 16 CPU threads; `results/gbdt/environment.json`
records the exact versions and devices.

## Protocol

The same rules as the NB/LR slot, so the numbers are directly comparable:

- **Data**: the shared cleaned splits only. Before fitting, the notebook checks
  all 21 partitions against the split card and checks that their ordered row-ID
  hashes equal the NB/LR run's (`results/gbdt/data_audit.csv`), i.e. both slots
  used byte-identical data.
- **Features**: word 1–2-gram TF-IDF on the reply, top 20,000 terms, fed to the
  trees as a sparse matrix (no SVD), plus the 17 reply-only surface counts left
  raw. Ablations add the 5 parent-context fields with a 5,000-term parent
  TF-IDF, or an out-of-fold subreddit target encoding (GroupKFold on
  `parent_key`). Everything is fitted on the training partition only.
- **Selection**: 25 Optuna (TPE, seed 3244) trials per library on the full
  `sarc_random` training partition, scored by validation ROC-AUC, with early
  stopping (learning rate 0.1, patience 100). The chosen hyper-parameters are
  frozen for every other fit.
- **Thresholds**: 0.50, plus the validation-chosen threshold that maximises
  recall at precision ≥ 0.75 (the team's operating point), frozen afterwards.
- **Stress schemes**: `sarc_temporal` and `sarc_subreddit` refit the features
  and booster on their own training partition with hyper-parameters, tree count
  and threshold frozen from `sarc_random`; nothing is selected on them.
- **Transfer**: the `sarc_random` models score the four non-SARC test sets with
  no target-side fitting or calibration.
- **Explanation**: on a fixed 10,000-row validation sample (test is never used),
  exact TreeSHAP for XGBoost and Saabas path attribution for LightGBM. Exact
  TreeSHAP overflows on LightGBM's comb-shaped trees (median depth 90); on
  XGBoost the two methods rank features with Spearman ρ = 0.983.

## Feature budget: the pilot

Fixed parameters, full `sarc_random` train, early stopping on validation:

| Vocabulary | Model | Histogram bins | Val ROC-AUC | Fit time | Peak GPU memory |
|---:|---|---:|---:|---:|---:|
| 10,000 | XGBoost (GPU) | 64 | 0.7960 | 116 s | 4.2 GB |
| 10,000 | LightGBM (CPU) | 64 | 0.7980 | 136 s | — |
| 20,000 | XGBoost (GPU) | 32 | 0.7990 | 119 s | 4.5 GB |
| 20,000 | XGBoost (GPU) | 64 | 0.7986 | 250 s | 6.8 GB |
| 20,000 | XGBoost (GPU) | 128 | 0.7982 | 2,254 s | 7.8 GB |
| 20,000 | LightGBM (CPU) | 64 | 0.8020 | 201 s | — |

Doubling the vocabulary to 20,000 helped both libraries. XGBoost gained nothing
from more histogram bins, while its GPU memory grew until it spilled into
shared system memory and slowed twenty-fold; a 50,000-term run crashed the
laptop. So XGBoost uses 32 bins and depth ≤ 8, its histogram cache is capped at
32 nodes (memory only: a fit with and without the cap gave bit-identical
margins), and a callback aborts any GPU fit that passes 7.2 GB. With these
settings the largest model, the parent-context ablation, peaked at 6.6 GB.

## Outputs

| Path | Contents |
|---|---|
| `results/gbdt/selected.json`, `tuning_trials_*.csv` | Chosen hyper-parameters and every Optuna trial |
| `results/gbdt/metrics.csv` | Every model × dataset × split × threshold, same columns as NB/LR |
| `results/gbdt/headline.csv`, `stress.csv`, `transfer.csv`, `ablation.csv` | Summary tables |
| `results/gbdt/slot_comparison.csv` | Against the NB/LR and glass-box slots on the same test set |
| `results/gbdt/attribution_*.csv` | Feature ranking, block shares, word directions, effect curves |
| `results/gbdt/attribution_check.json` | XGBoost TreeSHAP vs Saabas agreement |
| `results/gbdt/frozen.json`, `fit_log.json` | Frozen tree counts and thresholds; fit times and GPU memory peaks |
| `results/gbdt/ensemble_predictions/` | Person 6 handoff, described by `index.json` |
| `figures/gbdt/G01`–`G05` | ROC/PR, generalisation, ablations, top features, effect curves |
| `models/gbdt/` | Fitted boosters and feature builders (git-ignored, regenerated by training) |

The fitted boosters are saved in each library's native format
(`models/gbdt/<model id>.ubj` / `.txt`) next to the pickled `FeatureBuilder`
for their dataset and representation; `GBDT.load` plus `FeatureBuilder.transform`
reproduce every committed probability.

## Ensemble handoff

Same layout as `results/nb_logreg/ensemble_predictions/`: one parquet per
dataset and split, with `row_id` (BLAKE2b-128 of the cleaned reply text),
`row_position`, `label`, and one probability column per model named
`<training dataset>__<xgb|lgbm>_<representation>`. Join on `row_id`, check the
labels, and use one split scheme at a time. Choose ensemble weights on the
validation tables and keep test for the final evaluation.
`sarc_random__train_oof.parquet` holds 5-fold GroupKFold out-of-fold
predictions on the training partition for a stacking meta-model; full-train
predictions on the training rows must never be used in their place.
