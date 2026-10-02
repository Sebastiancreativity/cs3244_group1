#!/usr/bin/env python
"""
23_explain.py — read the model out loud.

This is the part of the project that only the glass-box slot can do. The other
base models need post-hoc attribution — permutation importance, SHAP — which
estimates what the model *probably* relies on. An EBM does not need estimating:
the fitted function is a sum of one curve per feature, so the curves *are* the
model, exactly and without approximation.

Three outputs:

  1. Global term importance — the EBM's own weighted mean absolute contribution
     per term, compared against the EDA's lift table where they overlap.
  2. Shape functions — the actual learned curve for the features that matter
     most. This is where the non-monotonic length effect either shows up or
     does not.
  3. The rule set — the mined conjunctions, printed verbatim with their
     coverage and the sarcasm rate inside them.

Usage
-----
    python modelling/glassbox/23_explain.py
    python modelling/glassbox/23_explain.py --top 24
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "pipeline"))
import config as C                                      # noqa: E402

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import joblib                                           # noqa: E402
import matplotlib                                       # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                         # noqa: E402
from common import viz                                  # noqa: E402

viz.apply_style()

#: Lift values from the EDA, for the overlap check.
EDA_LIFT = {"n_exclam": 2.811, "n_interjections": 1.971, "n_elongation": 1.485,
            "n_repeat_punct": 1.053, "n_ellipsis": 1.051,
            "n_allcaps_words": 1.051, "n_question": 0.885,
            "n_scare_quotes": 0.668}


def pretty(name: str) -> str:
    return name[4:] if name.startswith("has:") else name


# --------------------------------------------------------------------------
def fig_importance(df, out, top):
    d = df.head(top).iloc[::-1]
    fig, ax = plt.subplots(figsize=(9.4, 0.32 * len(d) + 1.8))
    y = np.arange(len(d))
    colors = [{"SURFACE": viz.SERIES[0], "LEXICAL": viz.SERIES[1],
               "LEXICON": viz.SERIES[2], "SUBREDDIT": viz.SERIES[3],
               "INTERACTION": viz.SERIES[6]}.get(b, viz.INK_MUTED)
              for b in d["block"]]
    ax.barh(y, d["importance"], height=0.66, color=colors)
    for i, r in enumerate(d.itertuples()):
        note = f"   EDA {EDA_LIFT[r.term]:.2f}x" if r.term in EDA_LIFT else ""
        ax.annotate(f"{r.importance:.4f}{note}", (r.importance, i),
                    xytext=(5, 0), textcoords="offset points", va="center",
                    fontsize=8, color=viz.INK_2)
    ax.set_yticks(y, [pretty(t) for t in d["term"]])
    ax.set_xlim(0, float(d["importance"].max()) * 1.45)
    ax.set_xlabel("Mean absolute contribution to the log-odds")
    ax.set_title("What the model uses, read off the model itself")
    viz.subtitle(ax, "EBM term importance — exact, not an attribution estimate")
    ax.grid(axis="y", visible=False)
    viz.despine(ax)

    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in
               [viz.SERIES[0], viz.SERIES[1], viz.SERIES[2], viz.SERIES[3]]]
    ax.legend(handles, ["surface", "lexical", "lexicon", "subreddit"],
              loc="lower right", fontsize=8.5)
    viz.save(fig, out, "term importance")


def fig_shapes(model, names, terms, out):
    """The learned curve for a handful of continuous features."""
    n = len(terms)
    ncol = 3
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.2 * ncol, 3.0 * nrow))
    axes = np.atleast_1d(axes).ravel()

    for ax, idx in zip(axes, terms):
        ti = model.term_features_.index((idx,))
        bins = model.bins_[idx][0]
        scores = np.asarray(model.term_scores_[ti])[1:-1]
        edges = np.asarray(bins, dtype=float)
        centres = np.concatenate([[edges[0]],
                                  (edges[:-1] + edges[1:]) / 2,
                                  [edges[-1]]])[:len(scores)]
        ax.step(centres, scores[:len(centres)], where="mid",
                color=viz.C_SARC, lw=2)
        ax.axhline(0, color=viz.INK_MUTED, lw=1, ls="--")
        ax.set_title(pretty(names[idx]), fontsize=10.5)
        ax.set_xlabel("feature value", fontsize=8.5)
        ax.set_ylabel("contribution to log-odds", fontsize=8.5)
        viz.despine(ax)
    for ax in axes[n:]:
        ax.set_visible(False)

    fig.suptitle("Shape functions — above zero pushes toward 'sarcastic'",
                 x=0.01, ha="left", fontsize=12, fontweight="600")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    viz.save(fig, out, "shape functions")


# --------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--top", type=int, default=26)
    args = ap.parse_args()

    meta = json.loads((C.CACHE / "features_done.json").read_text(encoding="utf-8"))
    names = meta["feature_names"]
    spans = meta["feature_groups"]
    sel = json.loads((C.RESULTS / "selection.json").read_text(encoding="utf-8"))

    def block_of(i):
        for b, (lo, hi) in spans.items():
            if lo <= i <= hi:
                return b
        return "RULE"

    # The EBM is the component that can be read exactly; if the winner is a
    # combination, its EBM half is the one explained.
    path = C.MODELS / ("ebm.joblib" if (C.MODELS / "ebm.joblib").exists()
                       else "glassbox_best.joblib")
    model = joblib.load(path)
    print(f"[load] explaining {path.name}  (winner was {sel['winner']})")

    expl = model.explain_global()
    data = expl.data()
    imp = pd.DataFrame({"term_idx": range(len(data["names"])),
                        "name": data["names"],
                        "importance": data["scores"]})

    rows = []
    for i, t in enumerate(model.term_features_):
        if len(t) == 1:
            rows.append({"term": names[t[0]], "block": block_of(t[0]),
                         "feature_idx": t[0],
                         "importance": float(imp.iloc[i]["importance"])})
        else:
            rows.append({"term": " x ".join(pretty(names[j]) for j in t),
                         "block": "INTERACTION", "feature_idx": -1,
                         "importance": float(imp.iloc[i]["importance"])})
    df = pd.DataFrame(rows).sort_values("importance", ascending=False)
    df.to_csv(C.RESULTS / "ebm_term_importance.csv", index=False)

    print(f"\n=== TOP {args.top} TERMS ===")
    for r in df.head(args.top).itertuples():
        note = f"   (EDA lift {EDA_LIFT[r.term]:.2f}x)" if r.term in EDA_LIFT else ""
        print(f"  {r.importance:.5f}  [{r.block:<11}] {pretty(r.term)}{note}")

    by_block = df.groupby("block")["importance"].sum()
    by_block = (by_block / by_block.sum()).sort_values(ascending=False)
    print("\n=== SHARE OF TOTAL CONTRIBUTION BY BLOCK ===")
    for b, v in by_block.items():
        print(f"  {b:<12} {v*100:5.1f}%")

    fig_importance(df, C.FIGURES / "G04_term_importance.png", args.top)

    # Shape functions for the most important continuous features.
    cont = [r.feature_idx for r in df.itertuples()
            if r.block in ("SURFACE", "LEXICON", "SUBREDDIT")
            and r.feature_idx >= 0][:9]
    if cont:
        fig_shapes(model, names, cont, C.FIGURES / "G05_shape_functions.png")

    # The rule set, with coverage measured on validation.
    bundle = joblib.load(C.MODELS / "mined_rules.joblib")
    Xva = np.load(C.CACHE / "X_val.npy")
    yva = np.load(C.CACHE / "y_val.npy")
    import importlib.util as ilu
    spec = ilu.spec_from_file_location("gb_train", HERE / "21_train.py")
    t = ilu.module_from_spec(spec)
    spec.loader.exec_module(t)

    rule_rows = []
    for r, txt in zip(bundle["rules"], bundle["text"]):
        mask = t.apply_rule(Xva, r)
        if not mask.any():
            continue
        rule_rows.append({"rule": txt, "coverage": round(float(mask.mean()), 4),
                          "sarcasm_rate": round(float(yva[mask].mean()), 4),
                          "n": int(mask.sum())})
    rdf = pd.DataFrame(rule_rows).sort_values("sarcasm_rate", ascending=False)
    rdf.to_csv(C.RESULTS / "rules.csv", index=False)

    print("\n=== STRONGEST SARCASM RULES (validation) ===")
    for r in rdf.head(8).itertuples():
        print(f"  {r.sarcasm_rate:.3f} sarcastic, covers {r.coverage*100:4.1f}%  "
              f"| {r.rule}")
    print("\n=== STRONGEST SINCERITY RULES ===")
    for r in rdf.tail(5).itertuples():
        print(f"  {r.sarcasm_rate:.3f} sarcastic, covers {r.coverage*100:4.1f}%  "
              f"| {r.rule}")

    (C.RESULTS / "explanation.json").write_text(json.dumps({
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "explained_model": path.name,
        "winner": sel["winner"],
        "n_terms": int(len(df)),
        "block_share": {k: round(float(v), 4) for k, v in by_block.items()},
        "top_terms": df.head(40).to_dict("records"),
        "n_rules": int(len(rdf)),
        "top_sarcasm_rules": rdf.head(12).to_dict("records"),
        "top_sincerity_rules": rdf.tail(8).to_dict("records"),
        "note": "EBM term importances are exact decompositions of the fitted "
                "model, not post-hoc attribution estimates. They describe the "
                "model, which is not the same as describing sarcasm.",
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[save] {C.RESULTS / 'explanation.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
