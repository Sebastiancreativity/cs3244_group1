#!/usr/bin/env python
"""
21_train.py — fit and compare every glass-box candidate.

Eight candidates, all of them models whose decision a person can read:

    decision_tree_d4   depth-limited CART. The readability floor — if a
                       four-level tree gets close to the rest, the extra
                       machinery is not earning its keep.
    figs               Fast Interpretable Greedy-tree Sums: a small sum of
                       shallow trees.
    boosted_rules      SLIPPER-style boosted rule set.
    skope_rules        rules filtered by precision and recall thresholds.
    rulefit            rules mined from trees, then an L1 linear layer over
                       them plus the raw features.
    ebm                Explainable Boosting Machine: a GAM with optional
                       pairwise interactions, fitted by cyclic boosting.
    ebm_plus_rules     EBM over the features plus mined rule indicators, so
                       conjunctions the GAM cannot express become inputs.
    ebm_rulefit_avg    probability average of the two strongest components.

Selection is on validation ROC-AUC. The test set is never read here.

Combinations and the glass-box constraint
-----------------------------------------
A combination stays glass-box only if a reader can still trace a prediction.
Both combinations here qualify: `ebm_plus_rules` is one additive model whose
inputs happen to include readable conjunctions, and `ebm_rulefit_avg` is the
mean of two models that can each be printed in full. A stack with a learned
meta-model over opaque bases would not qualify, which is why there isn't one.

Usage
-----
    python modelling/glassbox/21_train.py
    python modelling/glassbox/21_train.py --only ebm rulefit
    python modelling/glassbox/21_train.py --quick        # small grids, fast
    python modelling/glassbox/21_train.py --resume       # continue an interrupted sweep
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
import config as C                                      # noqa: E402

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import joblib                                           # noqa: E402
from sklearn.ensemble import GradientBoostingClassifier  # noqa: E402
from sklearn.metrics import (accuracy_score, f1_score, precision_score,  # noqa: E402
                             recall_score, roc_auc_score)
from sklearn.tree import DecisionTreeClassifier, _tree   # noqa: E402

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


# --------------------------------------------------------------------------
def load(tag):
    return (np.load(C.CACHE / f"X_{tag}.npy"),
            np.load(C.CACHE / f"y_{tag}.npy"))


def score(y, proba) -> dict:
    pred = (proba >= 0.5).astype(int)
    return {
        "roc_auc": round(float(roc_auc_score(y, proba)), 5),
        "accuracy": round(float(accuracy_score(y, pred)), 4),
        "precision": round(float(precision_score(y, pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y, pred, zero_division=0)), 4),
        "f1": round(float(f1_score(y, pred, zero_division=0)), 4),
    }


def proba_of(model, X) -> np.ndarray:
    """imodels estimators are inconsistent about predict_proba shape."""
    p = model.predict_proba(X)
    p = np.asarray(p)
    if p.ndim == 2 and p.shape[1] == 2:
        return p[:, 1]
    return p.ravel()


# --------------------------------------------------------------------------
# rule mining, shared by rulefit-style candidates
# --------------------------------------------------------------------------
def mine_rules(X, y, names, n_estimators, max_depth, max_rules, seed):
    """
    Extract short conjunctions from a shallow gradient-boosted ensemble.

    Each rule is a path from root to node, rendered as a readable predicate.
    Returned as (callables-free) threshold tuples so they can be applied to any
    matrix and printed verbatim in the report.
    """
    gb = GradientBoostingClassifier(n_estimators=n_estimators,
                                    max_depth=max_depth,
                                    learning_rate=0.1,
                                    subsample=0.7,
                                    random_state=seed)
    gb.fit(X, y)

    rules = []
    for est in gb.estimators_.ravel():
        tree = est.tree_

        def walk(node, conds):
            if tree.feature[node] == _tree.TREE_UNDEFINED:
                if conds:
                    rules.append(tuple(conds))
                return
            f = int(tree.feature[node])
            t = float(tree.threshold[node])
            walk(tree.children_left[node], conds + [(f, "<=", t)])
            walk(tree.children_right[node], conds + [(f, ">", t)])

        walk(0, [])

    # Deduplicate, then keep the rules that fire on a useful slice of the data:
    # a rule matching 0.1% of rows is noise, one matching 95% says nothing.
    seen, kept = set(), []
    for r in rules:
        key = tuple(sorted((f, o, round(t, 6)) for f, o, t in r))
        if key in seen:
            continue
        seen.add(key)
        mask = apply_rule(X, r)
        cov = mask.mean()
        if 0.01 <= cov <= 0.90:
            lift = abs(y[mask].mean() - y.mean()) if mask.any() else 0.0
            kept.append((r, cov, lift))

    kept.sort(key=lambda x: -x[2])
    kept = kept[:max_rules]
    return [r for r, _, _ in kept], [render_rule(r, names) for r, _, _ in kept]


def apply_rule(X, rule) -> np.ndarray:
    mask = np.ones(len(X), dtype=bool)
    for f, op, t in rule:
        mask &= (X[:, f] <= t) if op == "<=" else (X[:, f] > t)
    return mask


def render_rule(rule, names) -> str:
    parts = []
    for f, op, t in rule:
        n = names[f]
        if n.startswith("has:"):
            term = n[4:]
            parts.append(f"contains '{term}'" if op == ">" else f"no '{term}'")
        else:
            parts.append(f"{n} {op} {t:.3g}")
    return " AND ".join(parts)


def rule_matrix(X, rules) -> np.ndarray:
    out = np.zeros((len(X), len(rules)), dtype=np.float32)
    for j, r in enumerate(rules):
        out[:, j] = apply_rule(X, r)
    return out


# --------------------------------------------------------------------------
# candidate constructors
# --------------------------------------------------------------------------
def make(name, params, n_features):
    from imodels import (BoostedRulesClassifier, FIGSClassifier,
                         RuleFitClassifier, SkopeRulesClassifier)
    from interpret.glassbox import ExplainableBoostingClassifier

    if name == "decision_tree_d4":
        return DecisionTreeClassifier(random_state=C.SEED,
                                      min_samples_leaf=200, **params)
    if name == "figs":
        return FIGSClassifier(random_state=C.SEED, **params)
    if name == "boosted_rules":
        return BoostedRulesClassifier(random_state=C.SEED, **params)
    if name == "skope_rules":
        return SkopeRulesClassifier(random_state=C.SEED, **params)
    if name == "rulefit":
        return RuleFitClassifier(random_state=C.SEED, **params)
    if name in ("ebm", "ebm_plus_rules"):
        return ExplainableBoostingClassifier(**C.EBM_FIXED, **params)
    raise ValueError(name)


def fit_one(name, params, Xtr, ytr, Xva, yva, names, rules=None):
    t0 = time.time()
    if name == "ebm_plus_rules":
        Rtr, Rva = rule_matrix(Xtr, rules), rule_matrix(Xva, rules)
        Xtr2 = np.hstack([Xtr, Rtr])
        Xva2 = np.hstack([Xva, Rva])
        model = make("ebm", params, Xtr2.shape[1])
        model.fit(Xtr2, ytr)
        p = proba_of(model, Xva2)
    else:
        model = make(name, params, Xtr.shape[1])
        model.fit(Xtr, ytr)
        p = proba_of(model, Xva)
    return model, p, round(time.time() - t0, 1)


def complexity(name, model, params) -> str:
    """One short phrase describing how much there is to read."""
    try:
        if name == "decision_tree_d4":
            return f"{model.get_n_leaves()} leaves, depth {model.get_depth()}"
        if name == "ebm":
            n = len(model.term_features_)
            k = sum(1 for t in model.term_features_ if len(t) > 1)
            return f"{n} terms ({n - k} main effects, {k} interactions)"
        if name == "ebm_plus_rules":
            n = len(model.term_features_)
            return f"{n} terms, incl. mined rule indicators"
        if name == "rulefit":
            rs = getattr(model, "rules_", None)
            return f"{len(rs)} rules kept" if rs is not None else str(params)
        if name in ("boosted_rules", "skope_rules", "figs"):
            return str(params)
    except Exception:                                   # noqa: BLE001
        pass
    return str(params)


# --------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", nargs="*", default=None, choices=C.CANDIDATES)
    ap.add_argument("--quick", action="store_true",
                    help="first grid entry per candidate only")
    ap.add_argument("--resume", action="store_true",
                    help="skip trials already in candidate_trials_partial.csv")
    args = ap.parse_args()

    meta = json.loads((C.CACHE / "features_done.json").read_text(encoding="utf-8"))
    names = meta["feature_names"]

    Xtr, ytr = load("train")
    Xva, yva = load("val")
    print(f"[data] train={Xtr.shape}  val={Xva.shape}  "
          f"{len(names)} named features\n")

    print("[rules] mining conjunctions from a shallow boosted ensemble ...")
    t0 = time.time()
    rules, rule_text = mine_rules(Xtr, ytr, names, seed=C.SEED, **C.RULE_MINING)
    print(f"        {len(rules)} rules kept ({time.time()-t0:.0f}s)")
    for r in rule_text[:5]:
        print(f"        · {r}")
    joblib.dump({"rules": rules, "text": rule_text},
                C.MODELS / "mined_rules.joblib", compress=3)

    wanted = args.only or C.CANDIDATES
    trials, best_per = [], {}

    # Fits here run from seconds to tens of minutes. Every completed trial is
    # written to disk immediately, along with its validation probabilities and
    # the fitted estimator, so an interrupted sweep resumes instead of
    # restarting. --resume skips anything already on disk.
    partial = C.RESULTS / "candidate_trials_partial.csv"
    done_keys: set[tuple[str, str]] = set()
    if args.resume and partial.exists():
        prev = pd.read_csv(partial)
        trials = prev.to_dict("records")
        done_keys = {(r["model"], r["params"]) for r in trials
                     if not isinstance(r.get("error"), str)}
        for nm in {r["model"] for r in trials}:
            pp = C.CACHE / f"proba_val_{nm}.npy"
            mp = C.MODELS / f"cand_{nm}.joblib"
            if not pp.exists():
                continue
            rows = [r for r in trials if r["model"] == nm and "roc_auc" in r]
            if not rows:
                continue
            bst = max(rows, key=lambda r: r["roc_auc"])
            best_per[nm] = {
                "model": joblib.load(mp) if mp.exists() else None,
                "params": bst["params"],
                "score": {k: bst[k] for k in
                          ("roc_auc", "accuracy", "precision", "recall", "f1")},
                "proba": np.load(pp),
                "complexity": bst.get("complexity", ""),
                "seconds": bst.get("seconds", 0.0)}
        print(f"[resume] {len(done_keys)} trials already on disk; "
              f"{len(best_per)} candidates restored\n")

    for name in wanted:
        if name == "ebm_rulefit_avg":
            continue                                    # built after the rest
        grid = C.GRIDS.get(name, [{}])
        if args.quick:
            grid = grid[:1]
        print(f"\n=== {name} ===")
        for params in grid:
            if (name, str(params)) in done_keys:
                print(f"  {str(params):<52} (cached)")
                continue
            try:
                model, p, secs = fit_one(name, params, Xtr, ytr, Xva, yva,
                                         names, rules)
            except Exception as exc:                    # noqa: BLE001
                print(f"  {str(params):<52} FAILED: {type(exc).__name__}: {exc}")
                trials.append({"model": name, "params": str(params),
                               "error": f"{type(exc).__name__}: {exc}"})
                pd.DataFrame(trials).to_csv(partial, index=False)
                continue
            m = score(yva, p)
            cx = complexity(name, model, params)
            rec = {"model": name, "params": str(params), "seconds": secs,
                   "complexity": cx, **m}
            trials.append(rec)
            flag = ""
            if name not in best_per or m["roc_auc"] > best_per[name]["score"]["roc_auc"]:
                best_per[name] = {"model": model, "params": params, "score": m,
                                  "proba": p, "complexity": cx, "seconds": secs}
                np.save(C.CACHE / f"proba_val_{name}.npy", p)
                joblib.dump(model, C.MODELS / f"cand_{name}.joblib", compress=3)
                flag = "  <-- best"
            pd.DataFrame(trials).to_csv(partial, index=False)
            print(f"  {str(params):<52} AUC={m['roc_auc']:.5f} "
                  f"F1={m['f1']:.4f} {secs:>6.1f}s  [{cx}]{flag}", flush=True)

    # -- the averaging combination ----------------------------------------
    if (args.only is None or "ebm_rulefit_avg" in (args.only or [])) \
            and "ebm" in best_per and "rulefit" in best_per:
        print("\n=== ebm_rulefit_avg ===")
        p = 0.5 * best_per["ebm"]["proba"] + 0.5 * best_per["rulefit"]["proba"]
        m = score(yva, p)
        cx = (f"avg of [{best_per['ebm']['complexity']}] and "
              f"[{best_per['rulefit']['complexity']}]")
        trials.append({"model": "ebm_rulefit_avg", "params": "0.5/0.5",
                       "seconds": 0.0, "complexity": cx, **m})
        best_per["ebm_rulefit_avg"] = {"model": None, "params": {"weight": 0.5},
                                       "score": m, "proba": p,
                                       "complexity": cx, "seconds": 0.0}
        print(f"  {'0.5 EBM + 0.5 RuleFit':<52} AUC={m['roc_auc']:.5f} "
              f"F1={m['f1']:.4f}")

    # -- leaderboard -------------------------------------------------------
    board = sorted(({"model": k, **v["score"], "complexity": v["complexity"],
                     "seconds": v["seconds"], "params": str(v["params"])}
                    for k, v in best_per.items()),
                   key=lambda r: -r["roc_auc"])
    print("\n=== LEADERBOARD (validation) ===")
    print(f"  {'model':<20}{'AUC':>9}{'acc':>8}{'P':>8}{'R':>8}{'F1':>8}   complexity")
    for r in board:
        print(f"  {r['model']:<20}{r['roc_auc']:>9.5f}{r['accuracy']:>8.4f}"
              f"{r['precision']:>8.4f}{r['recall']:>8.4f}{r['f1']:>8.4f}   "
              f"{r['complexity']}")

    winner = board[0]["model"]
    print(f"\n[best] {winner} (validation AUC {board[0]['roc_auc']:.5f})")

    pd.DataFrame(trials).to_csv(C.RESULTS / "candidate_trials.csv", index=False)
    if best_per[winner]["model"] is not None:
        joblib.dump(best_per[winner]["model"],
                    C.MODELS / "glassbox_best.joblib", compress=3)
    for k in ("ebm", "rulefit"):
        if k in best_per and best_per[k]["model"] is not None:
            joblib.dump(best_per[k]["model"], C.MODELS / f"{k}.joblib", compress=3)
        elif (C.MODELS / f"cand_{k}.joblib").exists():
            joblib.dump(joblib.load(C.MODELS / f"cand_{k}.joblib"),
                        C.MODELS / f"{k}.joblib", compress=3)

    (C.RESULTS / "selection.json").write_text(json.dumps({
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "selection_metric": C.SELECTION_METRIC,
        "train_rows": int(len(ytr)), "val_rows": int(len(yva)),
        "n_features": len(names),
        "n_mined_rules": len(rules),
        "leaderboard": board,
        "winner": winner,
        "winner_params": str(best_per[winner]["params"]),
        "test_touched": False,
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[save] {C.RESULTS / 'selection.json'}")
    print(f"[save] {C.RESULTS / 'candidate_trials.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
