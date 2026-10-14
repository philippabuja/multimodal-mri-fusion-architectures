"""Per-scan preprocessing: HD-BET brain mask on T1, every other modality resampled into T1 space.

Output per scan (no normalisation, no masking — that happens at load time in 3_model/dataset.py):
    merged/T1.nii.gz, merged/<MOD>.nii.gz, merged/brain_mask.nii.gz, merged/channel_names.json
"""

import json
from pathlib import Path

import nibabel as nib
import numpy as np
from nibabel.processing import resample_from_to

from hdbet import run_hdbet


def _resample_to_t1(path: Path, t1_img, channel_idx: int | None = None) -> np.ndarray:
    """Resample via the NIfTI affines (scanner coordinates) into the T1 grid, trilinear."""
    img = nib.load(str(path))
    if channel_idx is not None:                       # pick one map out of a 4D file (R2*)
        img = nib.Nifti1Image(img.get_fdata(dtype=np.float32)[..., channel_idx], img.affine, img.header)
    return resample_from_to(img, t1_img, order=1, cval=0).get_fdata(dtype=np.float32)


def _save(data: np.ndarray, name: str, t1_img, out_dir: Path):
    hdr = t1_img.header.copy()
    hdr.set_data_dtype(np.float32)
    hdr["scl_slope"], hdr["scl_inter"] = 1.0, 0.0
    nib.save(nib.Nifti1Image(np.nan_to_num(data, nan=0., posinf=0., neginf=0.), t1_img.affine, hdr),
             str(out_dir / f"{name}.nii.gz"))


def process_scan(t1_path: Path, modalities: list[dict], output_dir: Path, device: str = "cuda"):
    """modalities: [{"name": "FLAIR", "path": ...}, {"name": "R2star", "path": ..., "channel_idx": 1}]"""
    merged, tmp = output_dir / "merged", output_dir / "temp"
    merged.mkdir(parents=True, exist_ok=True)
    tmp.mkdir(parents=True, exist_ok=True)
    if (merged / "brain_mask.nii.gz").exists() and (merged / "channel_names.json").exists():
        return

    run_hdbet(t1_path, tmp / "T1_brain.nii.gz", tmp / "brain_mask.nii.gz", device=device)
    nib.save(nib.load(str(tmp / "brain_mask.nii.gz")), str(merged / "brain_mask.nii.gz"))

    t1_img = nib.load(str(t1_path))
    _save(t1_img.get_fdata(dtype=np.float32), "T1", t1_img, merged)
    channels = ["T1"]

    for mod in modalities:
        if not Path(mod["path"]).exists():
            continue
        _save(_resample_to_t1(Path(mod["path"]), t1_img, mod.get("channel_idx")), mod["name"], t1_img, merged)
        channels.append(mod["name"])

    (merged / "channel_names.json").write_text(json.dumps({"channels": channels}, indent=2))
