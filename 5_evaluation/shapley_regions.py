"""Region-level Shapley values of the logit margin (Monte-Carlo permutation sampling).

Players = Harvard-Oxford regions + cerebellum + rest of brain, so they partition the brain mask.
A removed player is filled with the scan's mean in-brain intensity (per modality).
Efficiency check: sum of Shapley values == margin(full) - margin(all players removed).
Convergence check: Spearman rho between the values from the even and odd permutations.

    python 5_evaluation/shapley_regions.py --run-prefix FLAIR_s42_cv --permutations 200
    CUDA_VISIBLE_DEVICES=1 python 5_evaluation/shapley_regions.py ... --shard 1 --n-shards 2
Writes OUTPUT_DIR/analysis/shapley/{run_prefix}_shapley[_shardK].csv and _convergence.csv.
"""

import argparse
import csv
import json
import time
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
from scipy.stats import spearmanr

from common import (DEVICE, SPLITS_DIR, balanced_scans, brain_mask, fold_of, label_for,
                    load_fold_model, load_volume, margin, scan_dir)
from build_region_masks import NAMES_JSON
from config import ANALYSIS_DIR, region_mask_path


def build_players(scan, brain):
    lab = nib.load(str(region_mask_path("ho_labels", scan))).get_fdata().astype(np.int16) * brain
    names = {int(k): v for k, v in json.loads(NAMES_JSON.read_text()).items()}
    nxt = max(names) + 1
    cb = (nib.load(str(region_mask_path("cerebellum", scan))).get_fdata() > 0.5) & brain & (lab == 0)
    if cb.any():
        lab[cb] = nxt; names[nxt] = "Cerebellum"; nxt += 1
    rest = brain & (lab == 0)
    if rest.any():
        lab[rest] = nxt; names[nxt] = "Rest of brain"
    return lab, names


@torch.no_grad()
def shapley_for_scan(model, vol, brain, lab, ids, n_perm, batch, rng):
    C = vol.shape[0]
    base = torch.from_numpy(vol).to(DEVICE)
    fill = torch.tensor([float(vol[c][brain].mean()) for c in range(C)], device=DEVICE).view(C, 1, 1, 1)
    lab_t = torch.from_numpy(lab).to(DEVICE)
    masks = {v: lab_t == v for v in ids}
    any_player = torch.zeros_like(lab_t, dtype=torch.bool)
    for m in masks.values():
        any_player |= m

    m_full = margin(model(base[None])[0])[0].item()
    x_empty = torch.where(any_player[None], fill, base)
    m_empty = margin(model(x_empty[None])[0])[0].item()

    contrib = {v: [] for v in ids}
    for _ in range(n_perm):
        restored = torch.zeros_like(any_player)
        states, owners = [], []
        for i in rng.permutation(len(ids)):           # restore players one at a time
            restored = restored | masks[ids[i]]
            states.append(restored.clone())
            owners.append(ids[i])
        margins = []
        for s in range(0, len(states), batch):
            xb = torch.stack([torch.where(m[None], base, x_empty) for m in states[s:s + batch]])
            margins += margin(model(xb)[0]).cpu().tolist()
        prev = m_empty
        for owner, mg in zip(owners, margins):
            contrib[owner].append(mg - prev)
            prev = mg

    vals = {v: float(np.mean(contrib[v])) for v in ids}
    rho = spearmanr([np.mean(contrib[v][0::2]) for v in ids],
                    [np.mean(contrib[v][1::2]) for v in ids])[0]
    return vals, float(rho), m_full, m_empty


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-prefix", required=True)
    ap.add_argument("-n", "--n-scans", type=int, default=None, help="default: all scans")
    ap.add_argument("--permutations", type=int, default=200)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--n-shards", type=int, default=1)
    ap.add_argument("--cv-dir", type=Path, default=SPLITS_DIR)
    ap.add_argument("--n-folds", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    rng = np.random.default_rng(a.seed + a.shard)
    folds = fold_of(a.cv_dir, a.n_folds)
    scans = balanced_scans(folds, a.n_scans)[a.shard::a.n_shards]

    out_dir = ANALYSIS_DIR / "shapley"
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = f"_shard{a.shard}" if a.n_shards > 1 else ""
    fh = (out_dir / f"{a.run_prefix}_shapley{tag}.csv").open("w", newline="")
    fc = (out_dir / f"{a.run_prefix}_convergence{tag}.csv").open("w", newline="")
    w, wc = csv.writer(fh), csv.writer(fc)
    w.writerow(["scan_id", "label", "region", "shapley", "region_ml"])
    wc.writerow(["scan_id", "label", "margin_full", "margin_empty", "sum_shapley",
                 "efficiency_err", "convergence_rho", "seconds"])

    models = {}
    for i, scan in enumerate(scans, 1):
        k = folds[scan]
        if k not in models:
            models[k] = load_fold_model(a.run_prefix, k)
        if models[k] is None:
            continue
        model, cfg, norm_fns = models[k]
        merged = scan_dir(scan, cfg["data_root"])
        brain = brain_mask(merged)
        lab, names = build_players(scan, brain)
        ids = sorted(v for v in names if (lab == v).any())
        vol = load_volume(merged, cfg["modalities"], norm_fns)

        t = time.time()
        vals, rho, m_full, m_empty = shapley_for_scan(model, vol, brain, lab, ids,
                                                      a.permutations, a.batch, rng)
        tot = sum(vals.values())
        y = label_for(scan)
        for v in ids:
            w.writerow([scan, y, names[v], round(vals[v], 6), round(float((lab == v).sum()) * 0.008, 1)])
        wc.writerow([scan, y, round(m_full, 4), round(m_empty, 4), round(tot, 4),
                     round(abs(tot - (m_full - m_empty)), 6), round(rho, 4), round(time.time() - t, 1)])
        fh.flush(); fc.flush()
        print(f"[{i}/{len(scans)}] {scan}  sum={tot:+.3f}  rho={rho:+.3f}")
    fh.close(); fc.close()


if __name__ == "__main__":
    main()
