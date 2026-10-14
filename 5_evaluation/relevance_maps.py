"""Per-scan, per-modality DTD relevance maps of a CV run (out-of-fold, as in predict_oof.py).

    python 5_evaluation/relevance_maps.py --run-prefix FLAIR_R2star_MTR_dyn_ws_s42_cv

Volumes: RELEVANCE_ROOT/{run_prefix}/{scan}/rel_{MOD}.nii.gz (native grid, scan affine)
CSV:     OUTPUT_DIR/analysis/relevance/{run_prefix}.csv with, per scan, P(AD), the fraction of
         positive relevance inside the brain mask (vsrim) and each modality's share of it.
"""

import argparse
import csv
from pathlib import Path

import nibabel as nib
import numpy as np
import torch

from common import (DEVICE, SPLITS_DIR, brain_mask, fold_of, label_for, load_fold_model,
                    load_volume, scan_dir)
from config import ANALYSIS_DIR, relevance_dir


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-prefix", required=True)
    ap.add_argument("--cv-dir", type=Path, default=SPLITS_DIR)
    ap.add_argument("--n-folds", type=int, default=10)
    ap.add_argument("--no-volumes", action="store_true", help="write only the CSV")
    a = ap.parse_args()

    folds = fold_of(a.cv_dir, a.n_folds)
    rows = []
    for k in range(a.n_folds):
        loaded = load_fold_model(a.run_prefix, k)
        if loaded is None:
            continue
        model, cfg, norm_fns = loaded
        mods = cfg["modalities"]
        for scan in (s for s, f in folds.items() if f == k):
            merged = scan_dir(scan, cfg["data_root"])
            x = torch.from_numpy(load_volume(merged, mods, norm_fns)[None]).to(DEVICE)
            logits, heat = model(x)
            rel = heat[0].cpu().numpy().astype(np.float32)          # [M, X, Y, Z]
            mask = brain_mask(merged)

            pos = np.clip(rel, 0, None)
            inside = [float(p[mask].sum()) for p in pos]
            total = [float(p.sum()) for p in pos]
            row = {"scan_id": scan, "label": label_for(scan), "fold": k,
                   "p_ad": round(float(torch.softmax(logits, 1)[0, 1]), 6),
                   "vsrim": round(sum(inside) / (sum(total) + 1e-12), 6)}
            for m, mod in enumerate(mods):
                row[f"share_{mod}"] = round(inside[m] / (sum(inside) + 1e-12), 6)
            rows.append(row)

            if not a.no_volumes:
                ref = nib.load(str(merged / f"{mods[0]}.nii.gz"))
                dest = relevance_dir(a.run_prefix, scan)
                dest.mkdir(parents=True, exist_ok=True)
                for m, mod in enumerate(mods):
                    nib.save(nib.Nifti1Image(rel[m], ref.affine, ref.header), str(dest / f"rel_{mod}.nii.gz"))

    out = ANALYSIS_DIR / "relevance" / f"{a.run_prefix}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"{len(rows)} scans, mean vsrim {np.mean([r['vsrim'] for r in rows]):.4f} -> {out}")


if __name__ == "__main__":
    main()
