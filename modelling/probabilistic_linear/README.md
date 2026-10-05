# Person 1 — Notebook submission

Open **[person1_baselines.ipynb](person1_baselines.ipynb)**. It is the complete
Person 1 implementation and executed report. This folder contains only the
notebook, this README and `requirements.txt`; no separate model Python script
is required. The upload destination is **`python_nb_ver2`**.

This GitHub submission contains the notebook, report, figures, metrics and ensemble
prediction tables. Fitted model files under `artifacts/person1/` are intentionally
excluded and retained locally. The saved-model demo skips when they are absent.

## What the notebook contains

Part A defines data checks, reply-only surface features, full sparse word TF-IDF,
MNB/CNB/LR, validation-based tuning and threshold selection, final fits,
independent stress refits, frozen-model transfer, coefficient explanations,
permutation importance, plots/report generation and fitted model export.
It also contains the five executable protocol checks.
Part B displays the completed full-data experiment and optionally reruns it.

## Open and run

1. Place the delivered folders in the existing group repository with their
   relative paths preserved. The notebook uses the team's existing shared
   `pipeline/load.py`; the delivery is not a replacement for the whole repository.
2. Use Python 3.12 and install the recorded packages from the repository root:

   ```bash
   python -m pip install -r modelling/probabilistic_linear/requirements.txt
   ```

3. Open `person1_baselines.ipynb` and run all cells from the top.
   `RUN_TRAINING=False` reviews the completed results and runs the small protocol
   checks. The optional saved-model demo runs only when both fitted artifacts and
   the shared data are present.
4. To reproduce training, first obtain/rebuild the team's prepared splits under
   `data/splits/`, following the group's preprocessing workflow. Then set
   `RUN_TRAINING=True` and rerun all cells. This fits 20 tuning candidates and
   18 final models on the full appropriate training partitions. CPU is sufficient.

The notebook needs the team's prepared splits for fitting; the raw and cleaned
large datasets are not duplicated in this submission. During this run's data
reproduction, the original Python split-builder's path/list `CATALOG` name
collision was fixed locally without changing split membership. If reproducing
via that original Python script, alias its imported path as `CATALOG_PATH` and
use that name for the final catalogue output path. No team preprocessing file
is included or overwritten by this notebook-only delivery.

## Results and optional local models

- `../../docs/person1_report.md`: English report and limitations.
- `../../figures/person1/`: six figures displayed in the notebook.
- `../../results/person1/`: measured metrics, search trials, coefficients,
  data checks, environment versions and final verification.
- `../../artifacts/person1/`: optional local outputs, excluded from this GitHub
  submission. Rerun training and export to regenerate fitted models and transformers.
- `../../results/person1/ensemble_predictions/`: ten row-aligned probability
  tables for the Person 6 handoff, described by `index.json`.

After regenerating or separately obtaining the fitted artifacts and running
Part A in the notebook, its functions are available directly:

```python
bundle = joblib.load(ARTIFACTS / 'best_lr_pipeline.joblib')
sample = load_frame('sarc_random', 'test').iloc[:5]
probability = predict(bundle, sample)
other_probability = portable_predict('sarc_random__mnb_reply_tfidf', sample)
```

The compatibility cell in Part A maps the saved transformer's module name to
its notebook-defined class. Run it before loading model bundles; no adjacent
`experiment.py` is needed. Inputs are the shared CLEANED dataframes, including
surface columns when required. Load only trusted local joblib files.
All exported probabilities were verified against original estimators to 1e-10.

## Ensemble handoff

Prediction tables contain common `row_id`, `row_position`, `label` and a separate
probability column for each model. `row_id` is the BLAKE2b-128 digest of exact
cleaned target text. Join by row ID and validate labels rather than relying on
positions from independently sampled datasets. Choose voting weights using
validation predictions; reserve test predictions for final evaluation.

Use one split scheme at a time: the three SARC schemes overlap at corpus level.
MNB/CNB rankings coincide in this binary norm=False setting; they are not
independent ranking evidence. Their probabilities are not assumed calibrated.

`RUN_OOF=True` enables the optional additional GroupKFold stacking handoff.
The vectorizer, scaler and model are refitted inside each fold. OOF generation
has not been run for this delivery, and the Person 6 ensemble is not fitted here.
