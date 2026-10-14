"""Region occlusion per modality: replace one atlas region in ONE modality with that scan's mean
in-brain intensity and record the drop of the logit margin (AD - CN).

Rows with modality ALL_MODALITIES occlude the region in every channel at once.
Per-region AD - CN contrast = mean d_margin over AD scans minus mean over CN scans.

    python 5_evaluation/region_occlusion.py --run-prefix FLAIR_R2star_MTR_dyn_ws_s42_cv -n 120
Writes OUTPUT_DIR/analysis/occlusion/{run_prefix}.csv
"""

import argparse
import csv
import json
from pathlib import Path

import nibabel as nib
import numpy as np
import torch

from common import (DEVICE, SPLITS_DIR, balanced_scans, brain_mask, fold_of, label_for,
                    load_fold_model, load_volume, margin, scan_dir)
from build_region_masks import NAMES_JSON
from config import ANALYSIS_DIR, region_mask_path


@torch.no_grad()
def sweep_scan(model, vol, brain, region_masks, batch):
    """d[c, r] = margin(full) - margin(region r filled in modality c); last row = all modalities."""
    n_mod = vol.shape[0]
    x0 = torch.from_numpy(vol).to(DEVICE)
    m_full = margin(model(x0[None])[0])[0].item()
    fill = torch.tensor([float(vol[c][brain].mean()) for c in range(n_mod)],
                        device=DEVICE).view(n_mod, 1, 1, 1)
    jobs = [(c, r) for c in list(range(n_mod)) + [None] for r in range(len(region_masks))]
    masks = [torch.from_numpy(m).to(DEVICE) for m in region_masks]

    out = []
    for s in range(0, len(jobs), batch):
        chunk = jobs[s:s + batch]
        xb = x0.unsqueeze(0).repeat(len(chunk), 1, 1, 1, 1)
        for b, (c, r) in enumerate(chunk):
            if c is None:
                xb[b] = torch.where(masks[r][None], fill, xb[b])
            else:
                xb[b, c:c + 1] = torch.where(masks[r][None], fill[c], xb[b, c:c + 1])
        out += (m_full - margin(model(xb)[0])).cpu().tolist()
    return m_full, np.array(out).reshape(n_mod + 1, len(region_masks))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-prefix", required=True)
    ap.add_argument("-n", "--n-scans", type=int, default=None, help="default: all scans")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--cv-dir", type=Path, default=SPLITS_DIR)
    ap.add_argument("--n-folds", type=int, default=10)
    a = ap.parse_args()

    names = {int(k): v for k, v in json.loads(NAMES_JSON.read_text()).items()}
    ids = sorted(names)
    folds = fold_of(a.cv_dir, a.n_folds)
    models = {}

    out = ANALYSIS_DIR / "occlusion" / f"{a.run_prefix}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["scan_id", "label", "modality", "region", "region_ml", "d_margin", "margin_full"])
        for i, scan in enumerate(balanced_scans(folds, a.n_scans), 1):
            k = folds[scan]
            if k not in models:
                models[k] = load_fold_model(a.run_prefix, k)
            if models[k] is None:
                continue
            model, cfg, norm_fns = models[k]
            merged = scan_dir(scan, cfg["data_root"])
            brain = brain_mask(merged)
            lab = nib.load(str(region_mask_path("ho_labels", scan))).get_fdata().astype(np.int16) * brain
            region_masks = [(lab == v) & brain for v in ids]
            vol = load_volume(merged, cfg["modalities"], norm_fns)

            m_full, d = sweep_scan(model, vol, brain, region_masks, a.batch)
            mods = cfg["modalities"] + ["ALL_MODALITIES"]
            for c, mod in enumerate(mods):
                for r, v in enumerate(ids):
                    w.writerow([scan, label_for(scan), mod, names[v],
                                round(region_masks[r].sum() * 0.008, 2), round(d[c, r], 6), round(m_full, 6)])
            if i % 10 == 0:
                print(f"  {i} scans")
    print(f"-> {out}")


if __name__ == "__main__":
    main()
