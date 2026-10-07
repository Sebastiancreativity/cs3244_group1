"""Run the gradient-boosting notebook's stages from the command line.

The notebook is the single source of the code: this script executes Part A of
`gbdt_baselines.ipynb` (exactly as Jupyter would, in `__main__`) and then calls
one stage. Use it for the long runs, so each stage logs to the terminal, resumes
after an interruption, and can run in parallel with the other library:

    python modelling/gradient_boosting/run_stages.py audit
    python modelling/gradient_boosting/run_stages.py tune --families xgb     # GPU
    python modelling/gradient_boosting/run_stages.py tune --families lgbm    # CPU, in parallel
    python modelling/gradient_boosting/run_stages.py fit --families xgb --isolate
    python modelling/gradient_boosting/run_stages.py fit --families lgbm
    python modelling/gradient_boosting/run_stages.py explain
    python modelling/gradient_boosting/run_stages.py oof --families xgb --isolate
    python modelling/gradient_boosting/run_stages.py oof --families lgbm
    python modelling/gradient_boosting/run_stages.py export

`--isolate` runs every model (fit) or fold (oof) in its own child process, so
XGBoost's GPU memory cache is released in between; use it on GPUs with 8 GB.
Set the environment variable GBDT_THREADS to share the CPU between two runs.
Afterwards, open the notebook with RUN_TRAINING = False to view the results.
"""
import argparse
import json
import os
import subprocess
import sys
import time
import types
from pathlib import Path

import nbformat

HERE = Path(__file__).resolve().parent
NOTEBOOK = HERE / "gbdt_baselines.ipynb"

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("stage", choices=["audit", "tune", "fit", "merge", "explain", "oof", "export"])
ap.add_argument("--families", nargs="*", choices=["xgb", "lgbm"], default=None)
ap.add_argument("--trials", type=int, default=None, help="Optuna trials per library (default: notebook's N_TRIALS)")
ap.add_argument("--plan", nargs="*", default=None, help="fit only these dataset:representation items")
ap.add_argument("--folds", nargs="*", type=int, default=None, help="oof only these fold numbers")
ap.add_argument("--isolate", action="store_true", help="one child process per model (fit) or fold (oof)")
args = ap.parse_args()


def child(*extra):
    cmd = [sys.executable, "-u", str(Path(__file__)), *extra]
    if args.families:
        cmd += ["--families", *args.families]
    rc = subprocess.run(cmd).returncode
    if rc:
        sys.exit(f"child {' '.join(extra)} failed with exit code {rc}")


# Execute Part A in a module registered as __main__, so objects pickled here
# (the fitted feature builders) load in the notebook and vice versa.
os.chdir(HERE)
main = types.ModuleType("__main__")
sys.modules["__main__"] = main
for cell in nbformat.read(NOTEBOOK, as_version=4).cells:
    if cell.cell_type == "markdown" and cell.source.lstrip().startswith("## Part B"):
        break
    if cell.cell_type == "code":
        exec(compile(cell.source, str(NOTEBOOK), "exec"), main.__dict__)
ns = main.__dict__

t0 = time.time()
print(f"=== stage {args.stage} started {time.strftime('%H:%M:%S')} ===", flush=True)
fam = {"families": tuple(args.families)} if args.families else {}
plan = [tuple(p.split(":")) for p in args.plan] if args.plan else None
if args.stage == "audit":
    print(ns["setup"]())
    print(ns["audit_data"]().to_string())
elif args.stage == "tune":
    kw = dict(fam, **({"n_trials": args.trials} if args.trials else {}))
    print(json.dumps(ns["tune"](**kw), indent=2))
elif args.stage == "fit" and args.isolate:
    for d, r in plan or ns["run_plan"]():
        child("fit", "--plan", f"{d}:{r}")
    ns["merge_fit_outputs"]()
elif args.stage == "fit":
    selected = json.loads((ns["RESULTS"] / "selected.json").read_text())
    ns["fit_and_evaluate"](selected, plan=plan, **fam)
elif args.stage == "merge":
    print(ns["merge_fit_outputs"]().groupby(["training_dataset", "model"]).model_id.nunique().to_string())
elif args.stage == "explain":
    ns["explain"]()
elif args.stage == "oof" and args.isolate:
    for k in args.folds if args.folds is not None else range(5):
        child("oof", "--folds", str(k))
elif args.stage == "oof":
    ns["make_oof"](folds=args.folds, **fam)
elif args.stage == "export":
    tables = ns["summarise"](ns["merge_fit_outputs"]())
    print(ns["export_ensemble"]().to_string())
    ns["make_figures"](tables)
print(f"=== stage {args.stage} done in {(time.time() - t0) / 60:.1f} min ===", flush=True)
