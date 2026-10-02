#!/usr/bin/env python
"""
22_evaluate.py — score the selected glass-box model on the held-out test set.

Order of operations:

  1. read the winner chosen by 21_train.py on validation
  2. pick the operating threshold on VALIDATION, targeting precision rather
     than F1 (the proposal treats a false "sarcastic" call as the costly error)
  3. score sarc_random/test once
  4. apply the same model to the four transfer corpora
  5. compare against the other slots' reference numbers

Steps 4 and 5 are evaluation, not selection. Nothing about the model changes
because of a test or transfer score.

The stress splits (sarc_temporal, sarc_subreddit) are NOT scored here. They
partition the same corpus as sarc_random, so 70.1% of their test rows sit inside
this model's training data; comparing on them requires a retrain per scheme.

Usage
-----
    python modelling/glassbox/22_evaluate.py
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
from sklearn.metrics import (accuracy_score, average_precision_score,  # noqa: E402
                             confusion_matrix, f1_score, precision_recall_curve,
                             precision_score, recall_score, roc_auc_score,
                             roc_curve)
from common import viz                                  # noqa: E402

viz.apply_style()

sys.path.insert(0, str(HERE))
_train = __import__("importlib").import_module("importlib.util")
import importlib.util as _ilu                           # noqa: E402
_spec = _ilu.spec_from_file_location("gb_train", HERE / "21_train.py")
_t = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_t)


def load(tag):
    return (np.load(C.CACHE / f"X_{tag}.npy"),
            np.load(C.CACHE / f"y_{tag}.npy"))


def metrics(y, proba, thr) -> dict:
    pred = (proba >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "n": int(len(y)),
        "positive_rate": round(float(y.mean()), 4),
        "threshold": round(float(thr), 4),
        "accuracy": round(float(accuracy_score(y, pred)), 4),
        "precision": round(float(precision_score(y, pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y, pred, zero_division=0)), 4),
        "f1": round(float(f1_score(y, pred, zero_division=0)), 4),
        "roc_auc": round(float(roc_auc_score(y, proba)), 4),
        "pr_auc": round(float(average_precision_score(y, proba)), 4),
        "majority_baseline": round(float(max(y.mean(), 1 - y.mean())), 4),
        "confusion": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


def pick_threshold(y, proba, target_precision) -> dict:
    """
    Lowest threshold that reaches the target precision on validation.

    Lowest, not highest: among all points that satisfy the precision
    requirement, that is the one with the most recall. Sweeping for max F1
    instead would weight the two errors equally, which is not this project's
    cost structure.
    """
    prec, rec, thr = precision_recall_curve(y, proba)
    prec, rec = prec[:-1], rec[:-1]
    ok = np.where(prec >= target_precision)[0]
    if len(ok):
        i = int(ok[np.argmax(rec[ok])])
        rule = f"precision >= {target_precision}"
    else:                                               # unreachable: fall back
        f1 = np.divide(2 * prec * rec, prec + rec,
                       out=np.zeros_like(prec), where=(prec + rec) > 0)
        i = int(np.argmax(f1))
        rule = "max F1 (target precision unreachable)"

    grid = []
    for t in np.arange(0.20, 0.86, 0.05):
        p = (proba >= t).astype(int)
        grid.append({"threshold": round(float(t), 2),
                     "accuracy": round(float(accuracy_score(y, p)), 4),
                     "precision": round(float(precision_score(y, p, zero_division=0)), 4),
                     "recall": round(float(recall_score(y, p, zero_division=0)), 4),
                     "f1": round(float(f1_score(y, p, zero_division=0)), 4)})
    return {"rule": rule, "threshold": round(float(thr[i]), 4),
            "val_precision": round(float(prec[i]), 4),
            "val_recall": round(float(rec[i]), 4), "grid": grid}


_CACHE: dict = {}


def rebuild_proba(name, X, rules):
    """Reproduce any candidate's predictions on an arbitrary matrix."""
    if name == "ebm_rulefit_avg":
        return 0.5 * rebuild_proba("ebm", X, rules) \
             + 0.5 * rebuild_proba("rulefit", X, rules)
    if name not in _CACHE:
        _CACHE[name] = joblib.load(C.MODELS / f"cand_{name}.joblib")
    if name == "ebm_plus_rules":
        X = np.hstack([X, _t.rule_matrix(X, rules)])
    return _t.proba_of(_CACHE[name], X)


# --------------------------------------------------------------------------
def fig_curves(y, proba, out, label):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.3))
    fpr, tpr, _ = roc_curve(y, proba)
    auc = roc_auc_score(y, proba)
    ax1.plot(fpr, tpr, color=viz.C_SARC, label=f"{label} (AUC {auc:.3f})")
    ax1.plot([0, 1], [0, 1], color=viz.INK_MUTED, lw=1, ls="--", label="chance")
    ax1.set_xlabel("False positive rate"); ax1.set_ylabel("True positive rate")
    ax1.set_title("ROC — sarc_random test")
    viz.subtitle(ax1, "threshold-free ranking quality")
    ax1.legend(loc="lower right"); viz.despine(ax1)

    prec, rec, _ = precision_recall_curve(y, proba)
    ap = average_precision_score(y, proba)
    ax2.plot(rec, prec, color=viz.C_SARC, label=f"{label} (AP {ap:.3f})")
    ax2.axhline(float(y.mean()), color=viz.INK_MUTED, lw=1, ls="--",
                label=f"chance ({y.mean():.3f})")
    ax2.set_xlabel("Recall"); ax2.set_ylabel("Precision")
    ax2.set_title("Precision-recall — sarc_random test")
    viz.subtitle(ax2, "the operating point is chosen on this curve, on validation")
    ax2.legend(loc="lower left"); viz.despine(ax2)
    viz.save(fig, out, "ROC / PR")


def fig_leaderboard(board, out):
    d = pd.DataFrame(board).sort_values("roc_auc")
    fig, ax = plt.subplots(figsize=(9.6, 4.8))
    y = np.arange(len(d))
    colors = [viz.C_SARC if i == len(d) - 1 else viz.C_NOT for i in range(len(d))]
    ax.barh(y, d["roc_auc"], height=0.6, color=colors)
    for i, r in enumerate(d.itertuples()):
        ax.annotate(f"{r.roc_auc:.4f}", (r.roc_auc, i), xytext=(6, 0),
                    textcoords="offset points", va="center", fontsize=8.5,
                    color=viz.INK_2)
    ax.set_yticks(y, d["model"])
    ax.set_xlim(0.5, max(0.80, float(d["roc_auc"].max()) * 1.06))
    ax.axvline(0.5, color=viz.INK_MUTED, lw=1, ls="--")
    ax.set_xlabel("Validation ROC-AUC")
    ax.set_title("Glass-box candidates")
    viz.subtitle(ax, "all eight are models whose decisions can be read")
    ax.grid(axis="y", visible=False); viz.despine(ax)
    viz.save(fig, out, "leaderboard")


def fig_transfer(rows, out):
    d = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(9.6, 4.4))
    x = np.arange(len(d))
    b = ax.bar(x, d["roc_auc"], width=0.55,
               color=[viz.C_SARC if v >= 0.6 else viz.C_NOT for v in d["roc_auc"]])
    viz.bar_labels(ax, b, fmt="{:.3f}")
    ax.axhline(0.5, color=viz.INK, lw=1.2, ls="--")
    ax.annotate("chance", (len(d) - 0.45, 0.5), xytext=(4, 3),
                textcoords="offset points", fontsize=8, color=viz.INK_MUTED)
    ax.set_xticks(x, d["target"], rotation=20, ha="right")
    ax.set_ylabel("ROC-AUC"); ax.set_ylim(0.4, 0.85)
    ax.set_title("Transfer off Reddit")
    viz.subtitle(ax, "trained on SARC only; below the dashed line is worse than chance")
    ax.grid(axis="x", visible=False); viz.despine(ax)
    viz.save(fig, out, "transfer")


# --------------------------------------------------------------------------
def main() -> int:
    argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter).parse_args()

    sel = json.loads((C.RESULTS / "selection.json").read_text(encoding="utf-8"))
    winner = sel["winner"]
    board = sel["leaderboard"]
    rules = joblib.load(C.MODELS / "mined_rules.joblib")["rules"]

    # The winner beat plain EBM by 0.0004 AUC on validation, which is noise,
    # while carrying 70 extra terms. Both go to test so the report can argue
    # the simplicity trade-off from evidence rather than assertion.
    finalists = [winner] + [b["model"] for b in board[1:2] if b["model"] != winner]
    print(f"[load] winner = {winner} ({board[0]['complexity']})")
    print(f"[load] also evaluating runner-up = {finalists[-1]}"
          if len(finalists) > 1 else "")

    Xva, yva = load("val")
    Xte, yte = load("test")

    per_model, thr = {}, None
    for name in finalists:
        p_va = rebuild_proba(name, Xva, rules)
        t = pick_threshold(yva, p_va, C.TARGET_PRECISION)
        p_te = rebuild_proba(name, Xte, rules)
        per_model[name] = {"threshold": t,
                           "at_threshold": metrics(yte, p_te, t["threshold"]),
                           "at_0.50": metrics(yte, p_te, 0.5)}
        if name == winner:
            thr, p_te_win = t, p_te

    print("\n=== IN-DOMAIN (sarc_random test) ===")
    for name in finalists:
        r = per_model[name]
        print(f"  {name}")
        for tag, m in (("at chosen threshold", r["at_threshold"]),
                       ("at 0.50", r["at_0.50"])):
            print(f"    {tag:<22} acc={m['accuracy']:.4f} P={m['precision']:.4f} "
                  f"R={m['recall']:.4f} F1={m['f1']:.4f} AUC={m['roc_auc']:.4f}")
    if len(finalists) > 1:
        d = (per_model[finalists[0]]["at_0.50"]["roc_auc"]
             - per_model[finalists[1]]["at_0.50"]["roc_auc"])
        print(f"  --> test AUC gap between them: {d:+.5f}")
    ref = C.REFERENCE["lightgbm_person4"]
    print(f"  Person 4 LightGBM (black box)  acc={ref['accuracy']:.4f} "
          f"P={ref['precision']:.4f} R={ref['recall']:.4f} F1={ref['f1']:.4f} "
          f"AUC={ref['roc_auc']:.4f}")

    m_thr = per_model[winner]["at_threshold"]
    m_half = per_model[winner]["at_0.50"]
    p_te = p_te_win

    fig_curves(yte, p_te, C.FIGURES / "G01_roc_pr.png", winner)
    fig_leaderboard(board, C.FIGURES / "G02_leaderboard.png")

    print("\n=== TRANSFER (trained on SARC only) ===")
    rows = [{"target": "sarc_random (in-domain)", "roc_auc": m_half["roc_auc"],
             "accuracy": m_half["accuracy"], "f1": m_half["f1"],
             "majority": m_half["majority_baseline"]}]
    transfer = {}
    for name in C.TRANSFER_TARGETS:
        X = np.load(C.CACHE / f"X_{name}_test.npy")
        y = np.load(C.CACHE / f"y_{name}_test.npy")
        m = metrics(y, rebuild_proba(winner, X, rules), thr["threshold"])
        transfer[name] = m
        rows.append({"target": name, "roc_auc": m["roc_auc"],
                     "accuracy": m["accuracy"], "f1": m["f1"],
                     "majority": m["majority_baseline"]})
        print(f"  {name:<18} AUC={m['roc_auc']:.4f} acc={m['accuracy']:.4f} "
              f"F1={m['f1']:.4f}  (majority {m['majority_baseline']:.4f})")
    fig_transfer(rows, C.FIGURES / "G03_transfer.png")

    out = {
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "slot": "Person 5 — glass-box explainable (EBM + rule-based)",
        "winner": winner,
        "winner_complexity": board[0]["complexity"],
        "winner_params": sel["winner_params"],
        "n_features": sel["n_features"],
        "threshold": thr,
        "in_domain": {"at_threshold": m_thr, "at_0.50": m_half},
        "finalists": per_model,
        "transfer": transfer,
        "leaderboard": board,
        "reference": C.REFERENCE,
    }
    (C.RESULTS / "evaluation.json").write_text(
        json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    pd.DataFrame(rows).to_csv(C.RESULTS / "transfer_table.csv", index=False)
    print(f"\n[save] {C.RESULTS / 'evaluation.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
