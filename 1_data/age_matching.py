"""Age- and sex-matching of controls (ASPSF) to AD patients (ProDem) by propensity scores (PsmPy).

Input: one CSV with columns  id, class (0 = CN, 1 = AD), age, sex  — one row per scan folder that
has usable MRI data.  Output: matched_CN.json and matched_AD.json (lists of folder ids).

    python 1_data/age_matching.py --table cohort.csv --out splits/
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from psmpy import PsmPy


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--table", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()

    df = pd.read_csv(a.table)[["id", "class", "age", "sex"]].dropna()
    df["id"] = df["id"].astype(str).str.strip()
    print(f"before: CN={(df['class'] == 0).sum()}  AD={(df['class'] == 1).sum()}")

    psm = PsmPy(df, treatment="class", indx="id", exclude=[])
    psm.logistic_ps(balance=True)
    psm.kdtree_matched(matcher="propensity_logit", replacement=False, caliper=0.5,
                       drop_unmatched=True)
    ad = sorted(psm.matched_ids["id"].tolist())             # treatment group
    cn = sorted(psm.matched_ids["matched_ID"].tolist())     # matched controls

    for name, ids in (("CN", cn), ("AD", ad)):
        ages = df[df["id"].isin(ids)]["age"]
        print(f"after:  {name}={len(ids)}  age {ages.mean():.1f} ± {ages.std():.1f}")

    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "matched_CN.json").write_text(json.dumps(cn, indent=2))
    (a.out / "matched_AD.json").write_text(json.dumps(ad, indent=2))

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for ax, sub, title in ((axes[0], df, "before matching"),
                           (axes[1], df[df["id"].isin(cn + ad)], "after matching")):
        ax.hist(sub[sub["class"] == 0]["age"], bins=20, alpha=0.5, label="CN")
        ax.hist(sub[sub["class"] == 1]["age"], bins=20, alpha=0.5, label="AD")
        ax.set_title(title); ax.set_xlabel("age")
    axes[0].legend()
    fig.tight_layout()
    fig.savefig(a.out / "age_matching.png", dpi=150)


if __name__ == "__main__":
    main()
