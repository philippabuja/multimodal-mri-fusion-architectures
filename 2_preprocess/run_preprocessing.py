"""Preprocess every scan listed in a CSV into MRI_DATA_1MM/{study}/{folder}/merged/.

CSV columns: study, folder, T1 (path to the 1 mm T1), and one path column per modality
(FLAIR, R2star, MTR, ...; empty = not available). R2star files are 4D; map index 1 is used.

    python 2_preprocess/run_preprocessing.py --table scans.csv --gpus 0 1
"""

import argparse
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import data_root
from pipeline import process_scan

MODALITIES = {"FLAIR": None, "R2star": 1, "MTR": None}   # name -> channel index in a 4D file


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--table", type=Path, required=True)
    ap.add_argument("--gpus", nargs="+", default=["0"])
    a = ap.parse_args()

    df = pd.read_csv(a.table, dtype=str).fillna("")
    out_root = data_root("full")

    def job(args):
        i, row = args
        mods = [{"name": m, "path": row[m], **({"channel_idx": c} if c is not None else {})}
                for m, c in MODALITIES.items() if row.get(m)]
        try:
            process_scan(Path(row["T1"]), mods, out_root / row["study"] / row["folder"],
                         device=f"cuda:{a.gpus[i % len(a.gpus)]}")
            print(f"[{i + 1}/{len(df)}] {row['study']}/{row['folder']}")
        except Exception as e:
            print(f"[{i + 1}/{len(df)}] ERROR {row['study']}/{row['folder']}: {e}")

    with ThreadPoolExecutor(max_workers=len(a.gpus)) as ex:
        list(ex.map(job, enumerate(r for _, r in df.iterrows())))


if __name__ == "__main__":
    main()
