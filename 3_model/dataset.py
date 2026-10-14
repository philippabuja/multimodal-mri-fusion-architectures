from pathlib import Path

import nibabel as nib
import numpy as np
import torch
from scipy.ndimage import binary_erosion
from torch.utils.data import Dataset

# ------------------------------------------------------------------
# Intensity normalisation (applied per channel at load time).
# Fixed physical clips keep absolute values (R2*, MTR are quantitative maps);
# T1/FLAIR ceilings sit just above the in-brain maximum of this cohort.
# ------------------------------------------------------------------

def norm_t1_brain(d):  return np.clip(d, 0.0, 500.0) / 500.0
def norm_flair_lit(d): return np.clip(d, 0.0, 1100.0) / 1100.0
def norm_r2star_lit(d): return np.clip(d, 0.0, 150.0) / 150.0     # R2* in 1/s
def norm_mtr_lit(d):   return np.clip(d, 0.0, 1.0)                  # MTR is a ratio


NORM_FNS = {
    "none":       None,
    "t1_brain":   norm_t1_brain,
    "flair_lit":  norm_flair_lit,
    "r2star_lit": norm_r2star_lit,
    "mtr_lit":    norm_mtr_lit,
}

# Default normalisation per modality (as used in the thesis).
DEFAULT_NORM = {"T1": "t1_brain", "FLAIR": "flair_lit", "R2star": "r2star_lit", "MTR": "mtr_lit"}


def apply_norm(fn, data: np.ndarray) -> np.ndarray:
    return data if fn is None else fn(data)


def resolve_norm_fns(norm_spec, modalities) -> list:
    """norm_spec: None, one NORM_FNS key for all channels, or one key per modality."""
    n = len(modalities)
    if norm_spec is None:
        return [None] * n
    if isinstance(norm_spec, str):
        norm_spec = [norm_spec] * n
    if len(norm_spec) != n:
        raise ValueError(f"{len(norm_spec)} norm keys for {n} modalities {modalities}")
    return [NORM_FNS[k] for k in norm_spec]


def load_volume(merged_dir: Path, modalities, norm_fns) -> np.ndarray:
    """(C, X, Y, Z) float32 input exactly as the network sees it."""
    ch = []
    for mod, fn in zip(modalities, norm_fns):
        d = nib.load(str(merged_dir / f"{mod}.nii.gz")).get_fdata(dtype=np.float32)
        ch.append(apply_norm(fn, np.nan_to_num(d, nan=0.0, posinf=0.0, neginf=0.0)))
    return np.stack(ch, axis=0)


def erode(mask: np.ndarray, iterations: int) -> np.ndarray:
    """Binary brain mask for the RG loss, eroded by `iterations` voxels (0 = HD-BET boundary)."""
    if iterations <= 0:
        return (mask > 0.5).astype(np.float32)
    return binary_erosion(mask > 0.5, structure=np.ones((3, 3, 3)),
                          iterations=iterations).astype(np.float32)


class NiftiDataset(Dataset):
    """Returns (volume [C,X,Y,Z], rg_mask [1,X,Y,Z], label) for each scan directory.

    Expects {scan}/merged/{MOD}.nii.gz and {scan}/merged/brain_mask.nii.gz.
    Label: scans under the ProDem study folder are AD (1), all others CN (0).
    Scans with a missing modality or wrong shape are skipped.
    """

    def __init__(self, scan_dirs: list[Path], modalities: list[str], normalization_fn,
                 expected_shape: tuple, mask_erosion: int = 0):
        self.modalities = modalities
        self.norm_fns = normalization_fn
        self.mask_erosion = mask_erosion
        self.samples = []
        for d in scan_dirs:
            merged = d / "merged"
            files = [merged / f"{m}.nii.gz" for m in modalities] + [merged / "brain_mask.nii.gz"]
            if not all(f.exists() for f in files):
                print(f"[SKIP] {d.name}: file missing")
                continue
            if any(nib.load(str(f)).shape != expected_shape for f in files[:-1]):
                print(f"[SKIP] {d.name}: wrong shape")
                continue
            self.samples.append((merged, int(d.parent.name == "ProDem")))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        merged, label = self.samples[idx]
        vol = load_volume(merged, self.modalities, self.norm_fns)
        mask = nib.load(str(merged / "brain_mask.nii.gz")).get_fdata(dtype=np.float32)
        return (torch.from_numpy(vol),
                torch.from_numpy(erode(mask, self.mask_erosion)[np.newaxis]),
                label)
