"""Central path configuration. All locations come from environment variables (see README.md).

    MRI_DATA_1MM   preprocessed 1 mm volumes   {root}/{study}/{scan}/merged/*.nii.gz
    MRI_DATA_2MM   2 mm downsampled copy       (same layout)
    MRI_OUTPUT     checkpoints, CV summaries, analysis CSVs        (default: ./output)
    MRI_SPLITS     fold files fold_{k}.json                         (default: ./splits)
    MRI_WARPS      per-scan T1 -> MNI warps {study}__{scan}_warpcoef.nii.gz
    FSLDIR         FSL installation (MNI templates + atlases)
"""

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def _env(name: str, default=None) -> Path:
    val = os.environ.get(name, default)
    if val is None:
        raise RuntimeError(f"environment variable {name} is not set (see README.md)")
    return Path(val).expanduser()


DATA_DIR    = ROOT / "1_data"
PREPROC_DIR = ROOT / "2_preprocess"
MODEL_DIR   = ROOT / "3_model"
TRAIN_DIR   = ROOT / "4_train"
ENGINE_DIR  = TRAIN_DIR / "_engine"
EVAL_DIR    = ROOT / "5_evaluation"

OUTPUT_DIR   = _env("MRI_OUTPUT", ROOT / "output")
SPLITS_DIR   = _env("MRI_SPLITS", ROOT / "splits")
SUMMARY_DIR  = OUTPUT_DIR / "cv"
LOG_DIR      = OUTPUT_DIR / "cv_logs"
ANALYSIS_DIR = OUTPUT_DIR / "analysis"

# Per-scan relevance volumes and subject-space region masks are large; keep them beside the data.
RELEVANCE_ROOT   = OUTPUT_DIR / "relevance_volumes"
REGION_MASK_ROOT = OUTPUT_DIR / "region_masks"


def data_root(res: str) -> Path:
    """res = 'full' (1 mm, (176,224,256)) or 'downsampled' (2 mm, (88,112,128))."""
    return _env("MRI_DATA_1MM") if res == "full" else _env("MRI_DATA_2MM")


def fsl_dir() -> Path:
    return _env("FSLDIR")


def mni_dir() -> Path:
    return fsl_dir() / "data" / "standard"


def atlas_dir() -> Path:
    return fsl_dir() / "data" / "atlases"


def warp_dir() -> Path:
    return _env("MRI_WARPS", OUTPUT_DIR / "warps")


def run_dir(run: str) -> Path:
    return OUTPUT_DIR / run


def weights_dir(run: str) -> Path:
    return run_dir(run) / "weights"


def plots_dir(run: str) -> Path:
    return run_dir(run) / "plots"


def fold_run_name(prefix: str, k: int) -> str:
    """Fold k of a CV run lives in OUTPUT_DIR/{prefix}/fold{k}/."""
    return f"{prefix}/fold{k}"


def relevance_dir(run_prefix: str, scan: str) -> Path:
    return RELEVANCE_ROOT / run_prefix / scan


def region_mask_path(region: str, scan: str) -> Path:
    return REGION_MASK_ROOT / region / f"{scan}.nii.gz"
