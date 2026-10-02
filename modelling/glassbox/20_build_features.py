#!/usr/bin/env python
"""
20_build_features.py — build the interpretable feature matrix for Person 5.

Every column that comes out of this has a name a person can read off a chart.
Four groups:

  SURFACE (22)   engineered counts and ratios from the cleaning stage
  LEXICAL (300)  binary "the comment contains X" indicators for the 300 most
                 discriminative unigrams and bigrams
  LEXICON (16)   VADER and NRC EmoLex scores for the reply, the parent, and the
                 incongruity between them
  SUBREDDIT (2)  out-of-fold target encoding and log frequency

Selection and fitting happen on TRAIN ONLY. The term list, the encoding map and
the lexicon vocabularies never see validation, test or the transfer corpora.

Why the lexical terms are chosen by log-odds
--------------------------------------------
Ranking by raw frequency returns stopwords; ranking by a raw ratio returns
typos that occurred twice. The log-odds ratio with an informative Dirichlet
prior (Monroe, Colaresi & Quinn 2008) is the standard fix, and it is the same
estimator the EDA used, so the feature list is directly comparable to the
`yeah / because / obviously` list in the EDA report.

Why the lexicon block exists
----------------------------
Sentiment incongruity — a positive reply under a negative parent — is the most
cited engineered feature in the sarcasm literature, and Person 4's attribution
run recommended building it explicitly after finding that bag-of-words context
was worth almost nothing. This is that feature, in a form a GAM can show as a
single curve.

Usage
-----
    python modelling/glassbox/20_build_features.py
    python modelling/glassbox/20_build_features.py --force
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "pipeline"))

import config as C                                      # noqa: E402
from load import load_split, MODEL_SAFE_FEATURES, load_lexicon  # noqa: E402

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import joblib                                           # noqa: E402
from sklearn.model_selection import GroupKFold          # noqa: E402

RE_TOKEN = re.compile(r"[a-z][a-z']+")


# --------------------------------------------------------------------------
# lexical term selection
# --------------------------------------------------------------------------
def tokens(text: str) -> list[str]:
    return RE_TOKEN.findall(text.lower())


def bigrams(toks: list[str]) -> list[str]:
    return [f"{a}_{b}" for a, b in zip(toks, toks[1:])]


def log_odds_terms(texts_pos, texts_neg, n_uni, n_bi, min_count):
    """
    Monroe et al. (2008) log-odds ratio with an informative Dirichlet prior.

    Returns the n_uni unigrams and n_bi bigrams with the largest |z|, taking
    both directions — a term that strongly marks *sincerity* is as useful to a
    glass-box model as one that marks sarcasm.
    """
    cu_p, cu_n = Counter(), Counter()
    cb_p, cb_n = Counter(), Counter()
    for t in texts_pos:
        tk = tokens(t)
        cu_p.update(tk)
        cb_p.update(bigrams(tk))
    for t in texts_neg:
        tk = tokens(t)
        cu_n.update(tk)
        cb_n.update(bigrams(tk))

    def rank(cp, cn, k):
        prior = cp + cn
        vocab = [w for w in prior if prior[w] >= min_count]
        na, nb, n0 = sum(cp.values()), sum(cn.values()), sum(prior.values())
        a0 = n0 * 0.01
        scored = []
        for w in vocab:
            ya, yb, aw = cp[w], cn[w], prior[w] / n0 * a0
            oa = np.log((ya + aw) / (na + a0 - ya - aw))
            ob = np.log((yb + aw) / (nb + a0 - yb - aw))
            z = (oa - ob) / np.sqrt(1.0 / (ya + aw) + 1.0 / (yb + aw))
            scored.append((w, float(z)))
        scored.sort(key=lambda x: -abs(x[1]))
        return scored[:k]

    return rank(cu_p, cu_n, n_uni), rank(cb_p, cb_n, n_bi)


def lexical_matrix(texts: pd.Series, unigrams: list[str],
                   bigram_list: list[str]) -> np.ndarray:
    """Binary presence indicators. Presence, not count: 'contains yeah' is a
    statement a reader can check; 'yeah appears 1.7 times' is not."""
    uni_idx = {w: i for i, w in enumerate(unigrams)}
    bi_idx = {w: i + len(unigrams) for i, w in enumerate(bigram_list)}
    out = np.zeros((len(texts), len(unigrams) + len(bigram_list)), dtype=np.float32)
    for r, t in enumerate(texts.fillna("")):
        tk = tokens(t)
        for w in set(tk):
            j = uni_idx.get(w)
            if j is not None:
                out[r, j] = 1.0
        for w in set(bigrams(tk)):
            j = bi_idx.get(w)
            if j is not None:
                out[r, j] = 1.0
    return out


# --------------------------------------------------------------------------
# lexicon features
# --------------------------------------------------------------------------
class LexiconScorer:
    """VADER valence and NRC emotion counts, plus the incongruity terms."""

    EMOTIONS = ["anger", "anticipation", "disgust", "fear", "joy",
                "sadness", "surprise", "trust", "positive", "negative"]

    NAMES = ([f"nrc_{e}" for e in EMOTIONS] +
             ["vader_mean", "vader_pos_share", "vader_neg_share",
              "parent_vader_mean",
              "incongruity_pos_reply_neg_parent",
              "incongruity_signed"])

    def __init__(self):
        self.vader = load_lexicon("vader")
        nrc = load_lexicon("nrc")
        self.nrc = {e: nrc.get(e, set()) for e in self.EMOTIONS}

    def _vader_mean(self, toks: list[str]) -> tuple[float, float, float]:
        vals = [self.vader[w] for w in toks if w in self.vader]
        if not vals:
            return 0.0, 0.0, 0.0
        arr = np.asarray(vals, dtype=np.float32)
        return (float(arr.mean()),
                float((arr > 0).mean()),
                float((arr < 0).mean()))

    def transform(self, text: pd.Series, parent: pd.Series) -> np.ndarray:
        out = np.zeros((len(text), len(self.NAMES)), dtype=np.float32)
        n_emo = len(self.EMOTIONS)
        for r, (t, p) in enumerate(zip(text.fillna(""), parent.fillna(""))):
            tk = tokens(t)
            n = max(1, len(tk))
            s = set(tk)
            for j, e in enumerate(self.EMOTIONS):
                out[r, j] = len(s & self.nrc[e]) / n
            m, pos, neg = self._vader_mean(tk)
            out[r, n_emo] = m
            out[r, n_emo + 1] = pos
            out[r, n_emo + 2] = neg
            pm, _, _ = self._vader_mean(tokens(p))
            out[r, n_emo + 3] = pm
            # The classic cue: warm words in a reply to a hostile parent.
            out[r, n_emo + 4] = 1.0 if (m > 0.05 and pm < -0.05) else 0.0
            # Signed version, so a GAM can show the whole curve rather than a
            # single step. Negative = the two sides disagree in valence.
            out[r, n_emo + 5] = m * pm
        return out


# --------------------------------------------------------------------------
# subreddit encoding (identical scheme to the Person 4 run)
# --------------------------------------------------------------------------
class SubredditEncoder:
    def __init__(self, smoothing: float = 50.0, n_splits: int = 5):
        self.smoothing = smoothing
        self.n_splits = n_splits
        self.prior_ = 0.5
        self.mapping_: dict[str, float] = {}
        self.freq_: dict[str, int] = {}

    def _smooth(self, agg):
        n, mean = agg["count"], agg["mean"]
        return (mean * n + self.prior_ * self.smoothing) / (n + self.smoothing)

    def fit_transform_train(self, sub, y, groups) -> np.ndarray:
        self.prior_ = float(y.mean())
        oof = np.full(len(sub), self.prior_, dtype=np.float32)
        for tr, va in GroupKFold(n_splits=self.n_splits).split(sub, y, groups=groups):
            agg = (pd.DataFrame({"s": sub.iloc[tr].to_numpy(),
                                 "y": y.iloc[tr].to_numpy()})
                   .groupby("s")["y"].agg(["count", "mean"]))
            m = self._smooth(agg)
            oof[va] = sub.iloc[va].map(m).fillna(self.prior_).to_numpy("float32")
        full = (pd.DataFrame({"s": sub.to_numpy(), "y": y.to_numpy()})
                .groupby("s")["y"].agg(["count", "mean"]))
        self.mapping_ = self._smooth(full).to_dict()
        self.freq_ = full["count"].to_dict()
        return oof

    def transform(self, sub) -> np.ndarray:
        return sub.map(self.mapping_).fillna(self.prior_).to_numpy("float32")

    def frequency(self, sub) -> np.ndarray:
        return np.log1p(sub.map(self.freq_).fillna(0).to_numpy("float32"))


# --------------------------------------------------------------------------
def text_col(df):
    return "comment_clean" if "comment_clean" in df.columns else "text_clean"


def parent_col(df):
    for c in ("parent_clean", "context_clean"):
        if c in df.columns:
            return c
    return None


def build(df, unigrams, bigram_list, lex, enc, sub_te=None) -> np.ndarray:
    t = df[text_col(df)]
    pc = parent_col(df)
    p = df[pc] if pc else pd.Series([""] * len(df), index=df.index)

    surface = np.zeros((len(df), len(MODEL_SAFE_FEATURES)), dtype=np.float32)
    for j, c in enumerate(MODEL_SAFE_FEATURES):
        if c in df.columns:
            surface[:, j] = pd.to_numeric(df[c], errors="coerce") \
                              .astype("float32").fillna(0.0).to_numpy()

    lexical = lexical_matrix(t, unigrams, bigram_list)
    lexicon = lex.transform(t, p)

    if "subreddit" in df.columns:
        te = sub_te if sub_te is not None else enc.transform(df["subreddit"])
        freq = enc.frequency(df["subreddit"])
    else:
        te = np.full(len(df), enc.prior_, dtype=np.float32)
        freq = np.zeros(len(df), dtype=np.float32)
    sub = np.column_stack([te, freq]).astype(np.float32)

    return np.hstack([surface, lexical, lexicon, sub])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    marker = C.CACHE / "features_done.json"
    if marker.exists() and not args.force:
        print(f"cache present ({marker}); pass --force to rebuild")
        return 0

    t0 = time.time()
    train, val, test = load_split(C.TRAIN_SPLIT)
    print(f"[load] train={len(train):,} val={len(val):,} test={len(test):,}")

    rng = np.random.default_rng(C.SEED)
    if C.TRAIN_ROWS < len(train):
        idx = rng.choice(len(train), C.TRAIN_ROWS, replace=False)
        train = train.iloc[idx].reset_index(drop=True)
    if C.VAL_ROWS < len(val):
        idx = rng.choice(len(val), C.VAL_ROWS, replace=False)
        val = val.iloc[idx].reset_index(drop=True)
    print(f"[load] sampled to train={len(train):,} val={len(val):,}")

    print("[fit ] selecting lexical terms by log-odds (train only) ...")
    pos = train.loc[train.label == 1, "comment_clean"].tolist()
    neg = train.loc[train.label == 0, "comment_clean"].tolist()
    uni, bi = log_odds_terms(pos, neg, C.N_UNIGRAMS, C.N_BIGRAMS,
                             C.MIN_TERM_COUNT)
    unigrams = [w for w, _ in uni]
    bigram_list = [w for w, _ in bi]
    print(f"       {len(unigrams)} unigrams, {len(bigram_list)} bigrams")
    print("       top sarcastic:  " +
          ", ".join(w for w, z in uni if z > 0)[:120])
    print("       top sincere:    " +
          ", ".join(w for w, z in uni if z < 0)[:120])

    print("[fit ] lexicon scorer (VADER + NRC) ...")
    lex = LexiconScorer()

    print("[fit ] out-of-fold subreddit encoding ...")
    enc = SubredditEncoder()
    oof = enc.fit_transform_train(train["subreddit"], train["label"],
                                  train["parent_key"])

    names = (list(MODEL_SAFE_FEATURES)
             + [f"has:{w}" for w in unigrams]
             + [f"has:{w.replace('_', ' ')}" for w in bigram_list]
             + list(LexiconScorer.NAMES)
             + ["subreddit_rate", "subreddit_logfreq"])
    groups = {
        "SURFACE": [0, len(MODEL_SAFE_FEATURES) - 1],
        "LEXICAL": [len(MODEL_SAFE_FEATURES),
                    len(MODEL_SAFE_FEATURES) + len(unigrams) + len(bigram_list) - 1],
    }
    k = groups["LEXICAL"][1] + 1
    groups["LEXICON"] = [k, k + len(LexiconScorer.NAMES) - 1]
    groups["SUBREDDIT"] = [k + len(LexiconScorer.NAMES),
                           k + len(LexiconScorer.NAMES) + 1]
    print(f"[make] {len(names)} named features in {len(groups)} blocks")

    for tag, df, te in (("train", train, oof), ("val", val, None),
                        ("test", test, None)):
        X = build(df, unigrams, bigram_list, lex, enc, te)
        np.save(C.CACHE / f"X_{tag}.npy", X)
        np.save(C.CACHE / f"y_{tag}.npy", df["label"].to_numpy(dtype=np.int8))
        print(f"       {tag:<6} {X.shape}  {X.nbytes/1e6:.0f} MB")

    for name in C.TRANSFER_TARGETS:
        d = load_split(name, "test")
        X = build(d, unigrams, bigram_list, lex, enc)
        np.save(C.CACHE / f"X_{name}_test.npy", X)
        np.save(C.CACHE / f"y_{name}_test.npy", d["label"].to_numpy(dtype=np.int8))
        print(f"       {name:<18} {X.shape}")

    joblib.dump({"unigrams": unigrams, "bigrams": bigram_list,
                 "lexicon": lex, "encoder": enc, "names": names},
                C.MODELS / "glassbox_feature_pipeline.joblib", compress=3)

    marker.write_text(json.dumps({
        "built_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "elapsed_sec": round(time.time() - t0, 1),
        "n_features": len(names),
        "feature_names": names,
        "feature_groups": groups,
        "block_sizes": {k2: v[1] - v[0] + 1 for k2, v in groups.items()},
        "train_rows": int(len(train)),
        "val_rows": int(len(val)),
        "top_unigrams": [{"term": w, "z": round(z, 2)} for w, z in uni[:40]],
        "top_bigrams": [{"term": w, "z": round(z, 2)} for w, z in bi[:20]],
        "subreddit_prior": round(enc.prior_, 4),
    }, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\n[done] {time.time() - t0:.1f}s -> {C.CACHE}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
