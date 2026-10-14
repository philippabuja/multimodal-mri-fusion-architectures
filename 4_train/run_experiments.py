"""All thesis experiments: 10-fold patient-grouped CV on the T1+FLAIR+R2*+MTR common set.

    python 4_train/run_experiments.py single                   # 4 single-modality models
    python 4_train/run_experiments.py fusion                   # early + late fusion, 5 combinations
    python 4_train/run_experiments.py dyn                      # dynamic fusion, random init
    python 4_train/run_experiments.py dyn --warm-start         # dynamic fusion, encoders from 'single'
    python 4_train/run_experiments.py rg-ablation              # single models with RG loss off

Every mode runs seeds 42, 202 and 404 (--seeds to override). Run 'single' before 'dyn --warm-start'.
Decision-level fusion needs no training: see 5_evaluation/compare_models.py --decision.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import ENGINE_DIR
sys.path.insert(0, str(ENGINE_DIR))
from cv_runner import run_cv

SINGLES = {
    "T1":     (["T1"],     "t1_brain"),
    "FLAIR":  (["FLAIR"],  "flair_lit"),
    "R2star": (["R2star"], "r2star_lit"),
    "MTR":    (["MTR"],    "mtr_lit"),
}
COMBOS = ["FLAIR_R2star", "FLAIR_MTR", "R2star_MTR", "FLAIR_R2star_MTR", "T1_FLAIR_R2star_MTR"]


def combo(name):
    mods = name.split("_")
    return mods, [SINGLES[m][1] for m in mods]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["single", "fusion", "dyn", "rg-ablation"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[42, 202, 404])
    ap.add_argument("--warm-start", action="store_true")
    a = ap.parse_args()

    for seed in a.seeds:
        if a.mode == "single":
            for name, (mods, norm) in SINGLES.items():
                run_cv(mods, norm, f"{name}_s{seed}", seed=seed)

        elif a.mode == "rg-ablation":
            for name, (mods, norm) in SINGLES.items():
                run_cv(mods, norm, f"{name}_s{seed}_rgoff", seed=seed, rg_weight=0.0)

        elif a.mode == "fusion":
            for name in COMBOS:
                mods, norm = combo(name)
                run_cv(mods, norm, f"{name}_early_s{seed}", seed=seed, arch="early")
                run_cv(mods, norm, f"{name}_late_s{seed}", seed=seed, arch="late")

        elif a.mode == "dyn":
            warm = dict(warm_start_from="{mod}_s{seed}_cv", freeze_encoder_epochs=3,
                        encoder_lr_scale=0.3) if a.warm_start else {}
            for name in COMBOS:
                mods, norm = combo(name)
                run_cv(mods, norm, f"{name}_dyn{'_ws' if a.warm_start else ''}_s{seed}", seed=seed,
                       arch="dyn", rg_mode="per_modality", aux_weight=0.3,
                       weighting="confidence", modality_dropout=0.2, **warm)


if __name__ == "__main__":
    main()
