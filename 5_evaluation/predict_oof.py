"""Out-of-fold predictions of a CV run: every scan is scored by the fold model that never saw its patient.

    python 5_evaluation/predict_oof.py --run-prefix FLAIR_s42_cv
Writes OUTPUT_DIR/analysis/oof/{run_prefix}.csv  (scan_id, patient_id, label, p_ad, correct, fold).
"""

import argparse
import csv
from pathlib import Path

import torch

from common import (DEVICE, SPLITS_DIR, fold_of, label_for, load_fold_model, load_volume,
                    patient_id, scan_dir)
from config import ANALYSIS_DIR


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-prefix", required=True)
    ap.add_argument("--cv-dir", type=Path, default=SPLITS_DIR)
    ap.add_argument("--n-folds", type=int, default=10)
    a = ap.parse_args()

    folds = fold_of(a.cv_dir, a.n_folds)
    rows = []
    for k in range(a.n_folds):
        loaded = load_fold_model(a.run_prefix, k)
        if loaded is None:
            print(f"fold {k}: no checkpoint — skipped")
            continue
        model, cfg, norm_fns = loaded
        for scan in (s for s, f in folds.items() if f == k):
            merged = scan_dir(scan, cfg["data_root"])
            x = torch.from_numpy(load_volume(merged, cfg["modalities"], norm_fns)[None]).to(DEVICE)
            p = float(torch.softmax(model(x)[0], 1)[0, 1])
            y = label_for(scan)
            rows.append({"scan_id": scan, "patient_id": patient_id(scan), "label": y,
                         "p_ad": round(p, 6), "correct": int((p >= 0.5) == bool(y)), "fold": k})

    out = ANALYSIS_DIR / "oof" / f"{a.run_prefix}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"{len(rows)} scans, acc {sum(r['correct'] for r in rows) / len(rows):.4f} -> {out}")


if __name__ == "__main__":
    main()
