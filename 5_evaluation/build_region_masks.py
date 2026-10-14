"""Atlas regions pulled from MNI back into each scan's own 2 mm grid (inverse T1 -> MNI warp).

Occlusion and Shapley modify the network input, which lives in subject space, so the atlas has to
travel MNI -> subject. Labels: Harvard-Oxford cortical + subcortical (cerebral WM/cortex labels
dropped, subcortical overrides cortical) and the FSL cerebellum atlas. Nearest-neighbour
interpolation keeps labels intact; every mask is intersected with the scan's own brain mask.

    python 5_evaluation/build_region_masks.py
Writes REGION_MASK_ROOT/{ho_labels,cerebellum}/{scan}.nii.gz and ho_labels_names.json.
"""

import argparse
import json
import os
import subprocess
import tempfile
import xml.etree.ElementTree as ET
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import nibabel as nib
import numpy as np

from common import SPLITS_DIR, brain_mask, fold_of, scan_dir
from config import REGION_MASK_ROOT, atlas_dir, mni_dir, region_mask_path
from warp_to_mni import fsl_env, warp_path

DROP_SUB = {"left cerebral white matter", "right cerebral white matter",
            "left cerebral cortex", "right cerebral cortex"}
CEREBELLUM = "Cerebellum/Cerebellum-MNIfnirt-maxprob-thr25-2mm.nii.gz"
NAMES_JSON = REGION_MASK_ROOT / "ho_labels_names.json"


def build_ho_labels() -> Path:
    """Combined Harvard-Oxford label volume in MNI 2 mm + {label: name}."""
    out = REGION_MASK_ROOT / "ho_labels_MNI.nii.gz"
    if out.exists() and NAMES_JSON.exists():
        return out

    def read(xml, nii):
        names = {int(l.get("index")) + 1: l.text.strip()
                 for l in ET.parse(atlas_dir() / xml).getroot().iter("label")}
        return names, nib.load(str(atlas_dir() / nii)).get_fdata().astype(np.int16)

    cn, cort = read("HarvardOxford-Cortical.xml", "HarvardOxford/HarvardOxford-cort-maxprob-thr25-2mm.nii.gz")
    sn, sub = read("HarvardOxford-Subcortical.xml", "HarvardOxford/HarvardOxford-sub-maxprob-thr25-2mm.nii.gz")
    lab, names, nxt = np.zeros(cort.shape, np.int16), {}, 1
    for vol, nm, drop in ((cort, cn, set()), (sub, sn, DROP_SUB)):
        for v, n in sorted(nm.items()):
            if n.lower() in drop or not (vol == v).any():
                continue
            lab[vol == v] = nxt; names[nxt] = n; nxt += 1
    ref = nib.load(str(mni_dir() / "MNI152_T1_2mm_brain_mask.nii.gz"))
    REGION_MASK_ROOT.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(lab, ref.affine), str(out))
    NAMES_JSON.write_text(json.dumps({str(k): v for k, v in names.items()}, indent=1))
    return out


def build_scan(job):
    scan, sources = job
    merged = scan_dir(scan)
    brain = brain_mask(merged)
    env = fsl_env()
    with tempfile.TemporaryDirectory() as td:
        inv = f"{td}/inv"
        subprocess.run(["invwarp", f"--warp={warp_path(scan)}", f"--ref={merged / 'T1.nii.gz'}",
                        f"--out={inv}"], env=env, check=True, capture_output=True)
        for region, src in sources.items():
            tmp = f"{td}/{region}"
            subprocess.run(["applywarp", f"--in={src}", f"--ref={merged / 'T1.nii.gz'}",
                            f"--warp={inv}", "--interp=nn", f"--out={tmp}"],
                           env=env, check=True, capture_output=True)
            img = nib.load(tmp + ".nii.gz")
            lab = np.nan_to_num(img.get_fdata())
            data = (lab * brain).astype(np.int16) if region == "ho_labels" \
                else ((lab > 0) & brain).astype(np.uint8)
            out = region_mask_path(region, scan)
            out.parent.mkdir(parents=True, exist_ok=True)
            nib.save(nib.Nifti1Image(data, img.affine), str(out))
    return scan


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cv-dir", type=Path, default=SPLITS_DIR)
    ap.add_argument("--n-folds", type=int, default=10)
    ap.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 4) - 2))
    a = ap.parse_args()

    sources = {"ho_labels": str(build_ho_labels()), "cerebellum": str(atlas_dir() / CEREBELLUM)}
    scans = [s for s in sorted(fold_of(a.cv_dir, a.n_folds))
             if not all(region_mask_path(r, s).exists() for r in sources)]
    print(f"{len(scans)} scans to build")
    with ProcessPoolExecutor(max_workers=a.jobs) as ex:
        list(ex.map(build_scan, [(s, sources) for s in scans]))


if __name__ == "__main__":
    main()
