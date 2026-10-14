"""Patient-grouped, class-stratified k-fold splits on the scans that have ALL requested modalities.

Every model of the thesis (single modalities and all fusion variants) is trained and evaluated on
the same folds: 10 folds over the T1+FLAIR+R2star+MTR common set.

    python 1_data/make_cv_folds.py --cn matched_CN.json --ad matched_AD.json
Writes SPLITS_DIR/fold_{k}.json = {"training": [...], "validation": [...]}.
"""

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import SPLITS_DIR, data_root
from labels import collect_valid, patient_id


def grouped_kfold(scans: list[str], n_folds: int, seed: int) -> list[list[str]]:
    """Assign whole patients to folds, greedily balancing the number of scans per fold."""
    by_patient = defaultdict(list)
    for s in scans:
        by_patient[patient_id(s)].append(s)
    patients = list(by_patient)
    random.Random(seed).shuffle(patients)
    patients.sort(key=lambda p: len(by_patient[p]), reverse=True)

    folds, sizes = [[] for _ in range(n_folds)], [0] * n_folds
    for p in patients:
        f = min(range(n_folds), key=lambda i: sizes[i])
        folds[f].append(p)
        sizes[f] += len(by_patient[p])
    return folds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cn", type=Path, required=True, help="JSON list of matched CN folders (ASPSF)")
    ap.add_argument("--ad", type=Path, required=True, help="JSON list of matched AD folders (ProDem)")
    ap.add_argument("--modalities", nargs="+", default=["T1", "FLAIR", "R2star", "MTR"])
    ap.add_argument("--n-folds", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=Path, default=SPLITS_DIR)
    a = ap.parse_args()

    root = data_root("downsampled")

    def has_all(scan):
        m = root / scan / "merged"
        return (m / "brain_mask.nii.gz").exists() and \
            all((m / f"{x}.nii.gz").exists() for x in a.modalities)

    cn = [s for s in collect_valid(a.cn, "ASPSF") if has_all(s)]
    ad = [s for s in collect_valid(a.ad, "ProDem") if has_all(s)]
    print(f"common set ({'+'.join(a.modalities)}): CN={len(cn)}  AD={len(ad)}")

    by_patient = defaultdict(list)
    for s in cn + ad:
        by_patient[patient_id(s)].append(s)

    # stratify: fold each class separately, then merge fold i of CN with fold i of AD
    cn_f = grouped_kfold(cn, a.n_folds, a.seed)
    ad_f = grouped_kfold(ad, a.n_folds, a.seed + 1)
    fold_pids = [cn_f[i] + ad_f[i] for i in range(a.n_folds)]

    a.out.mkdir(parents=True, exist_ok=True)
    for i in range(a.n_folds):
        val_p = set(fold_pids[i])
        tr_p = set().union(*(fold_pids[j] for j in range(a.n_folds) if j != i))
        assert not (val_p & tr_p), f"fold {i}: patient overlap"
        val = sorted(s for p in val_p for s in by_patient[p])
        tr = sorted(s for p in tr_p for s in by_patient[p])
        rng = random.Random(a.seed + i)
        rng.shuffle(tr)
        rng.shuffle(val)
        (a.out / f"fold_{i}.json").write_text(json.dumps({"training": tr, "validation": val}, indent=2))
        n_ad = sum(s.startswith("ProDem/") for s in val)
        print(f"fold {i}: val {len(val)} (AD {n_ad} / CN {len(val) - n_ad})  train {len(tr)}")


if __name__ == "__main__":
    main()
