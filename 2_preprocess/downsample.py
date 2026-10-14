"""Downsample every preprocessed scan from 1 mm (176,224,256) to 2 mm isotropic (88,112,128).

Only T1 is acquired at 1 mm; FLAIR, R2* and MTR have 2-3 mm slices, so 2 mm is the working grid.
Images: trilinear; brain mask: nearest neighbour (stays binary).

    python 2_preprocess/downsample.py --workers 8
"""

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy.ndimage import zoom

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import data_root

SRC_SHAPE = (176, 224, 256)
MASK = "brain_mask.nii.gz"


def _resample(img, order: int):
    data = img.get_fdata(dtype=np.float32)
    pad = [(max(0, t - c) // 2, max(0, t - c) - max(0, t - c) // 2) for c, t in zip(data.shape, SRC_SHAPE)]
    small = zoom(np.pad(data, pad), 0.5, order=order)
    aff = img.affine.copy()
    aff[:3, :3] *= 2.0
    return nib.Nifti1Image(small.astype(np.float32), aff)


def _process(merged: Path, src: Path, dst: Path) -> str:
    out = dst / merged.relative_to(src)
    if (out / "channel_names.json").exists():
        return "skip"
    out.mkdir(parents=True, exist_ok=True)
    written = []
    for f in sorted(merged.glob("*.nii.gz")):
        img = _resample(nib.load(str(f)), order=0 if f.name == MASK else 1)
        if f.name == MASK:
            img = nib.Nifti1Image((img.get_fdata() > 0.5).astype(np.float32), img.affine)
        else:
            written.append(f.name[:-len(".nii.gz")])
        nib.save(img, str(out / f.name))
    (out / "channel_names.json").write_text(json.dumps({"channels": written}, indent=2))
    return "done"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()
    src, dst = data_root("full"), data_root("downsampled")
    dirs = sorted(p for p in src.rglob("merged") if p.is_dir())
    print(f"{len(dirs)} scans: {src} -> {dst}")
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        futs = [ex.submit(_process, d, src, dst) for d in dirs]
        for i, f in enumerate(as_completed(futs), 1):
            f.result()
            if i % 50 == 0 or i == len(dirs):
                print(f"  {i}/{len(dirs)}")


if __name__ == "__main__":
    main()
