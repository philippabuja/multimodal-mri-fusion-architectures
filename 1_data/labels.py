"""Label and patient-grouping conventions shared by fold building, training and evaluation.

Scan ids are relative paths '{study}/{patient}__{date}':
    ASPSF/*   -> cognitively normal (CN, label 0)
    ProDem/*  -> Alzheimer's disease (AD, label 1)
The patient key is the folder name before '__'; all scans of a patient stay on one side of a split.
"""

import json
import sys
from pathlib import Path

import nibabel as nib

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import SPLITS_DIR, data_root

EXPECTED_SHAPE = (176, 224, 256)


def excluded_patients() -> set[str]:
    """Patient ids to drop (e.g. failed brain extraction), one per line in SPLITS_DIR/exclude.txt."""
    f = SPLITS_DIR / "exclude.txt"
    return {ln.strip() for ln in f.read_text().splitlines() if ln.strip()} if f.exists() else set()


def patient_id(scan: str) -> str:
    return Path(scan).name.split("__")[0]


def label_for(scan: str) -> int:
    return 1 if scan.startswith("ProDem/") else 0


def collect_valid(json_path: Path, study: str) -> list[str]:
    """Scans of one study whose T1 + brain mask exist on the 1 mm grid and are not excluded."""
    exclude = excluded_patients()
    root = data_root("full")
    valid = []
    for folder in json.loads(Path(json_path).read_text()):
        scan = f"{study}/{folder}"
        if patient_id(scan) in exclude:
            continue
        merged = root / scan / "merged"
        t1, mask = merged / "T1.nii.gz", merged / "brain_mask.nii.gz"
        if t1.exists() and mask.exists() and nib.load(str(t1)).shape == EXPECTED_SHAPE:
            valid.append(scan)
    return valid
