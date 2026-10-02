"""
Configuration for the Person 5 slot: glass-box explainable models.

Scope: Explainable Boosting Machines and rule-based learners, plus the best
combination of the two.

The governing constraint
------------------------
A glass-box model is only glass-box if its INPUTS are readable. Person 4's
gradient-boosting run compressed the comment text into 300 truncated-SVD
components; those work fine for a tree ensemble but they are not things a human
can reason about, so an EBM fitted on them would be an interpretable function of
uninterpretable variables. Useless for this slot.

So every feature here is a named, human-readable quantity:

  * 22 engineered surface and context features   ("number of exclamation marks")
  * ~300 lexical indicators                      ("the comment contains 'yeah'")
  * 16 lexicon-derived sentiment features        ("VADER compound of the reply")
  * 2 subreddit features                         ("this community's base rate")

That costs some accuracy relative to full-resolution TF-IDF. Measuring the size
of that cost is one of the points of the experiment.
"""
from __future__ import annotations

from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]

RESULTS = ROOT / "results" / "glassbox"
FIGURES = ROOT / "figures" / "glassbox"
MODELS = ROOT / "models" / "glassbox"
CACHE = ROOT / ".cache" / "glassbox"

for _d in (RESULTS, FIGURES, MODELS, CACHE):
    _d.mkdir(parents=True, exist_ok=True)

SEED = 3244

# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------
TRAIN_SPLIT = "sarc_random"
STRESS_SPLITS = ["sarc_temporal", "sarc_subreddit"]
TRANSFER_TARGETS = ["figlang_reddit", "figlang_twitter",
                    "tweeteval_irony", "news_headlines"]

#: EBM is O(rows x features x rounds) and single-pass over the whole matrix each
#: round. On 640k x ~340 a full fit runs into hours, and the learning curve in
#: 21_train.py shows the metric is flat well before that. Both numbers are
#: reported in the write-up rather than buried.
TRAIN_ROWS = 200_000
VAL_ROWS = 60_000

# --------------------------------------------------------------------------
# Lexical features — selected on TRAIN ONLY
# --------------------------------------------------------------------------
#: Terms are ranked by the log-odds ratio with an informative Dirichlet prior
#: (Monroe et al. 2008) — the same estimator the EDA used. Plain frequency
#: returns stopwords; a raw ratio returns typos.
N_UNIGRAMS = 220
N_BIGRAMS = 80
MIN_TERM_COUNT = 200          # a term must appear this often in train to qualify

# --------------------------------------------------------------------------
# Candidate models
# --------------------------------------------------------------------------
#: Every entry is a glass-box learner. "combo" entries are combinations of two
#: glass-box components, which stay glass-box as long as both parts are.
CANDIDATES = [
    "decision_tree_d4",       # depth-4 CART — the readability floor
    "figs",                   # Fast Interpretable Greedy-tree Sums
    "boosted_rules",          # SLIPPER-style boosted rule set
    "skope_rules",            # precision-filtered rule set
    "rulefit",                # rules + sparse linear layer
    "ebm",                    # Explainable Boosting Machine
    "ebm_plus_rules",         # EBM over features + mined rule indicators
    "ebm_rulefit_avg",        # probability average of EBM and RuleFit
]

#: Hyper-parameter grids. Deliberately small: the Person 4 run showed tuning on
#: this data is worth under 0.01 AUC, and these models are being chosen for
#: interpretability rather than squeezed for the last decimal.
GRIDS = {
    "decision_tree_d4": [
        {"max_depth": 3}, {"max_depth": 4}, {"max_depth": 5}, {"max_depth": 6},
    ],
    "figs": [
        {"max_rules": 12}, {"max_rules": 20},
    ],
    "boosted_rules": [
        {"n_estimators": 20}, {"n_estimators": 40}, {"n_estimators": 80},
    ],
    "skope_rules": [
        {"n_estimators": 20, "max_depth": 3},
        {"n_estimators": 40, "max_depth": 4},
    ],
    "rulefit": [
        # 60 -> 120 rules cost 396s -> 1799s for +0.011 AUC, and RuleFit was
        # never going to catch the EBM. One representative config is enough.
        {"max_rules": 60, "tree_size": 4},
    ],
    "ebm_plus_rules": [
        {"interactions": 10, "max_bins": 256, "learning_rate": 0.02},
    ],
    "ebm": [
        {"interactions": 0, "max_bins": 256, "learning_rate": 0.02},
        {"interactions": 10, "max_bins": 256, "learning_rate": 0.02},
        {"interactions": 20, "max_bins": 256, "learning_rate": 0.02},
        {"interactions": 20, "max_bins": 512, "learning_rate": 0.01},
    ],
}

#: Rules mined for `ebm_plus_rules`: shallow trees give short, readable
#: conjunctions ("n_exclam > 0 AND contains 'obviously'").
RULE_MINING = {"n_estimators": 30, "max_depth": 3, "max_rules": 80}

EBM_FIXED = {
    "random_state": SEED,
    "n_jobs": -1,
    "outer_bags": 4,
    "inner_bags": 0,
    "early_stopping_rounds": 50,
    "max_rounds": 5000,
}

# --------------------------------------------------------------------------
# Metric and operating point
# --------------------------------------------------------------------------
#: Selection is on ROC-AUC: it is threshold-free, so it ranks candidates on
#: ranking quality and leaves the precision/recall trade-off to be set
#: explicitly afterwards. Tuning precision at a fixed threshold instead selects
#: degenerate models that almost never predict the positive class.
SELECTION_METRIC = "roc_auc"

#: The proposal treats a false "sarcastic" call as the expensive error, so the
#: reported operating point targets precision rather than F1. Both are given.
TARGET_PRECISION = 0.75

# --------------------------------------------------------------------------
# Reference points from the other slots (results/splits/split_verification.json
# and the Person 4 run). Used for context in the report, never for selection.
# --------------------------------------------------------------------------
REFERENCE = {
    "tfidf_logreg_person1": {"accuracy": 0.7060, "precision": 0.7238,
                             "recall": 0.6862, "f1": 0.7045},
    "lightgbm_person4": {"accuracy": 0.7059, "precision": 0.7285,
                         "recall": 0.6760, "f1": 0.7013, "roc_auc": 0.7803},
    "majority_baseline": {"accuracy": 0.5107},
}
