"""Warp per-scan relevance maps into MNI152 (2 mm) for group analysis.

Each scan needs a nonlinear T1 -> MNI warp in MRI_WARPS/{study}__{scan}_warpcoef.nii.gz.
Missing warps are computed with --build (FLIRT 12 dof on the brain + FNIRT on the whole head).
Because every modality was resampled into the scan's T1 grid, one warp covers all of them.

Warped maps are renormalised so the positive relevance inside the MNI brain mask sums to 1:
interpolation does not conserve the sum, and every scan should weigh equally in a group mean.

    python 5_evaluation/warp_to_mni.py --build --run-prefix FLAIR_s42_cv
    python 5_evaluation/warp_to_mni.py --run-prefix FLAIR_s42_cv
"""

import argparse
import os
import subprocess
import tempfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import nibabel as nib
import numpy as np

from common import SPLITS_DIR, fold_of, scan_dir
from config import RELEVANCE_ROOT, fsl_dir, mni_dir, warp_dir


def fsl_env():
    return os.environ | {"FSLDIR": str(fsl_dir()), "FSLOUTPUTTYPE": "NIFTI_GZ",
                         "PATH": f"{fsl_dir()}/bin:" + os.environ.get("PATH", "")}


def warp_path(scan: str) -> Path:
    return warp_dir() / f"{scan.replace('/', '__')}_warpcoef.nii.gz"


def build_warp(scan: str) -> str:
    out = warp_path(scan)
    if out.exists():
        return "exists"
    merged = scan_dir(scan, "full")
    out.parent.mkdir(parents=True, exist_ok=True)
    env, std = fsl_env(), mni_dir()
    with tempfile.TemporaryDirectory() as td:
        brain, aff = f"{td}/t1_brain", f"{td}/aff.mat"
        for cmd in (["fslmaths", merged / "T1.nii.gz", "-mas", merged / "brain_mask.nii.gz", brain],
                    ["flirt", "-in", brain, "-ref", std / "MNI152_T1_2mm_brain.nii.gz",
                     "-omat", aff, "-dof", "12"],
                    ["fnirt", f"--in={merged / 'T1.nii.gz'}", f"--aff={aff}",
                     f"--ref={std / 'MNI152_T1_2mm.nii.gz'}", "--config=T1_2_MNI152_2mm",
                     f"--cout={str(out)[:-len('.nii.gz')]}"]):
            subprocess.run([str(c) for c in cmd], env=env, check=True, capture_output=True)
    return "ok"


def warp_scan(job):
    run_prefix, scan = job
    src = RELEVANCE_ROOT / run_prefix / scan
    ref = mni_dir() / "MNI152_T1_2mm.nii.gz"
    mni_mask = nib.load(str(mni_dir() / "MNI152_T1_2mm_brain_mask.nii.gz")).get_fdata() > 0.5
    for f in sorted(src.glob("rel_*.nii.gz")):
        if f.name.endswith("_MNI.nii.gz"):
            continue
        out = src / f.name.replace(".nii.gz", "_MNI.nii.gz")
        subprocess.run(["applywarp", f"--in={f}", f"--ref={ref}", f"--warp={warp_path(scan)}",
                        "--interp=trilinear", f"--out={out}"], env=fsl_env(), check=True)
        img = nib.load(str(out))
        d = np.clip(np.nan_to_num(img.get_fdata(dtype=np.float32)), 0, None)
        s = float(d[mni_mask].sum())
        nib.save(nib.Nifti1Image((d / s if s > 0 else d).astype(np.float32), img.affine), str(out))
    return scan


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-prefix")
    ap.add_argument("--build", action="store_true", help="compute missing T1 -> MNI warps")
    ap.add_argument("--cv-dir", type=Path, default=SPLITS_DIR)
    ap.add_argument("--n-folds", type=int, default=10)
    ap.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 4) - 2))
    a = ap.parse_args()

    scans = sorted(fold_of(a.cv_dir, a.n_folds))
    with ProcessPoolExecutor(max_workers=a.jobs) as ex:
        if a.build:
            todo = [s for s in scans if not warp_path(s).exists()]
            print(f"building {len(todo)} warps")
            list(ex.map(build_warp, todo))
        if a.run_prefix:
            todo = [(a.run_prefix, s) for s in scans if (RELEVANCE_ROOT / a.run_prefix / s).exists()]
            print(f"warping {len(todo)} scans of {a.run_prefix}")
            list(ex.map(warp_scan, todo))


if __name__ == "__main__":
    main()
