"""Shared helpers: rebuild a fold's trained model from its config.json, load scans, OOF assignment."""

import json
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from config import MODEL_DIR, DATA_DIR, SPLITS_DIR, weights_dir, fold_run_name, data_root
sys.path.insert(0, str(MODEL_DIR))
sys.path.insert(0, str(DATA_DIR))
from model import build_model                     # noqa: E402
from dataset import resolve_norm_fns, load_volume  # noqa: E402
from labels import label_for, patient_id          # noqa: E402

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
N_BLOCKS = {"full": 5, "downsampled": 4}


def load_fold_model(prefix: str, k: int):
    """(model, cfg, norm_fns) for fold k of a CV run, or None if no checkpoint exists."""
    wdir = weights_dir(fold_run_name(prefix, k))
    if not (wdir / "best_model.pth").exists():
        return None
    cfg = json.loads((wdir / "config.json").read_text())
    kw = {k_: cfg[k_] for k_ in ("weighting", "modality_dropout", "dropout_mode")
          if cfg.get("arch") == "dyn" and cfg.get(k_) is not None}
    model = build_model(cfg["arch"], in_channels=cfg["in_channels"],
                        n_blocks=N_BLOCKS[cfg["data_root"]], with_dtd=True,
                        width=cfg.get("width", 8), **kw).to(DEVICE)
    model.load_state_dict(torch.load(wdir / "best_model.pth", map_location=DEVICE), strict=False)
    model.eval()
    return model, cfg, resolve_norm_fns(cfg["norm"], cfg["modalities"])


def fold_of(cv_dir: Path = SPLITS_DIR, n_folds: int = 10) -> dict:
    """{scan_id: fold whose validation set contains it} — each scan is scored by an unseen model."""
    out = {}
    for k in range(n_folds):
        for s in json.loads((cv_dir / f"fold_{k}.json").read_text())["validation"]:
            out[s] = k
    return out


def scan_dir(scan: str, res: str = "downsampled") -> Path:
    return data_root(res) / scan / "merged"


def brain_mask(merged: Path) -> np.ndarray:
    return nib.load(str(merged / "brain_mask.nii.gz")).get_fdata() > 0.5


def margin(logits):
    """Logit margin AD - CN. Used instead of P(AD) because the softmax saturates."""
    return logits[:, 1] - logits[:, 0]


def balanced_scans(folds: dict, n: int | None = None) -> list[str]:
    """AD and CN scans interleaved, so any prefix of the list is class-balanced."""
    ad = [s for s in sorted(folds) if label_for(s) == 1]
    cn = [s for s in sorted(folds) if label_for(s) == 0]
    if n:
        ad, cn = ad[:n - n // 2], cn[:n // 2]
    out = []
    for i in range(max(len(ad), len(cn))):
        out += ad[i:i + 1] + cn[i:i + 1]
    return out
