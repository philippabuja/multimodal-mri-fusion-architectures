"""Patient-grouped k-fold cross-validation: one train.py process per fold, scheduled over GPUs.

Fold k is trained on SPLITS_DIR/fold_k.json into OUTPUT_DIR/{prefix}/fold{k}/weights/.
A summary (per-fold and pooled accuracy with Wilson 95 % CI) goes to OUTPUT_DIR/cv/{label}.json.
"""

import json
import os
import queue
import subprocess
import sys
import threading
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config import SPLITS_DIR, SUMMARY_DIR, LOG_DIR, weights_dir, fold_run_name

TRAIN_PY = Path(__file__).resolve().parent / "train.py"
GPUS = [int(g) for g in os.environ.get("MRI_GPUS", "0").split(",")]
N_FOLDS = 10

# thesis recipe (2 mm data, 4-block network)
DEFAULTS = dict(data_root="downsampled", epochs=40, early_stopping_patience=12, max_attempts=3,
                seed=42, arch="early", rg_erosion=0)


def log(msg):
    print(msg, flush=True)


def _train_one(gpu, fold, run_name, mods, norm, cv_dir, **opts):
    cmd = [sys.executable, str(TRAIN_PY), "--modalities", *mods,
           "--norm", *([norm] if isinstance(norm, str) else norm),
           "--split", str(cv_dir / f"fold_{fold}.json"), "--run-name", run_name]
    for k, v in opts.items():
        if v is not None:
            cmd += [f"--{k.replace('_', '-')}", str(v)]
    env = os.environ | {"CUDA_VISIBLE_DEVICES": str(gpu)}
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log(f"  [GPU{gpu}] start {run_name}")
    with open(LOG_DIR / f"{run_name.replace('/', '_')}.log", "w") as fh:
        subprocess.run(cmd, env=env, stdout=fh, stderr=subprocess.STDOUT, check=False)
    log(f"  [GPU{gpu}] done  {run_name}")


def _run_jobs(jobs, gpus):
    q = queue.Queue()
    for j in jobs:
        q.put(j)

    def worker(gpu):
        while True:
            try:
                j = q.get_nowait()
            except queue.Empty:
                return
            _train_one(gpu=gpu, **j)

    threads = [threading.Thread(target=worker, args=(g,)) for g in gpus]
    for t in threads:
        t.start()
    for t in threads:
        t.join()


def _wilson(k, n, z=1.96):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / d
    return c - h, c + h


def summarize(prefix, label, cv_dir, n_folds):
    folds, accs, correct, n_tot = [], [], 0, 0
    for k in range(n_folds):
        cfg_p = weights_dir(fold_run_name(prefix, k)) / "config.json"
        if not cfg_p.exists():
            continue
        cfg = json.loads(cfg_p.read_text())
        folds.append({"fold": k, "acc": cfg["best_val_acc"], "auc": cfg["best_val_auc"],
                      "vsrim": cfg["best_val_vsrim"], "collapsed": cfg["collapsed"]})
        if cfg["best_val_acc"] is not None and not cfg["collapsed"]:
            accs.append(cfg["best_val_acc"])
            n_val = len(json.loads((cv_dir / f"fold_{k}.json").read_text())["validation"])
            correct += round(cfg["best_val_acc"] * n_val)
            n_tot += n_val
    summ = {"label": label, "prefix": prefix, "n_folds_ok": len(accs),
            "mean_acc": float(np.mean(accs)) if accs else None,
            "std_acc": float(np.std(accs)) if accs else None,
            "pooled_acc": correct / n_tot if n_tot else None,
            "wilson95": _wilson(correct, n_tot) if n_tot else None,
            "per_fold": folds}
    SUMMARY_DIR.mkdir(parents=True, exist_ok=True)
    (SUMMARY_DIR / f"{label}.json").write_text(json.dumps(summ, indent=2))
    if accs:
        log(f"=== {label}: {len(accs)}/{n_folds} folds  acc {summ['mean_acc']:.4f} ± {summ['std_acc']:.4f}"
            f"  pooled {summ['pooled_acc']:.4f}")
    return summ


def run_cv(mods, norm, label, prefix=None, cv_dir=SPLITS_DIR, n_folds=N_FOLDS, gpus=None, **opts):
    """Train all folds of one configuration, then write its summary.

    opts are forwarded to train.py (e.g. arch="dyn", rg_mode="per_modality", aux_weight=0.3).
    """
    prefix = prefix or f"{label}_cv"
    opts = DEFAULTS | opts
    log(f"\nCV {label}: {'+'.join(mods)}  {opts}")
    jobs = [dict(fold=k, run_name=fold_run_name(prefix, k), mods=mods, norm=norm,
                 cv_dir=cv_dir, **opts) for k in range(n_folds)]
    _run_jobs(jobs, gpus or GPUS)
    return summarize(prefix, label, cv_dir, n_folds)
