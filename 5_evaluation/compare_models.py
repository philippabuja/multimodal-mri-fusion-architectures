"""Compare CV runs on their out-of-fold predictions (paired on identical scans).

Reports accuracy, AUC, sensitivity, specificity per run and, against a reference run,
DeLong's test (AUC) and the exact McNemar test (accuracy).
--decision adds decision-level fusion: the mean P(AD) of the given single-modality runs.

    python 5_evaluation/compare_models.py --ref FLAIR_s42_cv \
        --runs FLAIR_R2star_early_s42_cv FLAIR_R2star_late_s42_cv FLAIR_R2star_dyn_ws_s42_cv \
        --decision FLAIR_s42_cv R2star_s42_cv
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import ANALYSIS_DIR

OOF_DIR = ANALYSIS_DIR / "oof"


def _midrank(x):
    order = np.argsort(x)
    z = x[order]
    n = len(x)
    t = np.zeros(n)
    i = 0
    while i < n:
        j = i
        while j < n and z[j] == z[i]:
            j += 1
        t[i:j] = 0.5 * (i + j - 1) + 1
        i = j
    out = np.empty(n)
    out[order] = t
    return out


def delong(p1, p2, y):
    """Two-sided DeLong test for two correlated AUCs. Returns (auc1, auc2, p)."""
    order = np.argsort(-y, kind="mergesort")
    probs, y = np.vstack([p1, p2])[:, order], y[order]
    m = int(y.sum()); n = len(y) - m
    tx = np.vstack([_midrank(r[:m]) for r in probs])
    ty = np.vstack([_midrank(r[m:]) for r in probs])
    tz = np.vstack([_midrank(r) for r in probs])
    aucs = tz[:, :m].sum(1) / m / n - (m + 1.0) / 2.0 / n
    v01 = (tz[:, :m] - tx) / n
    v10 = 1.0 - (tz[:, m:] - ty) / m
    cov = np.cov(v01) / m + np.cov(v10) / n
    var = cov[0, 0] + cov[1, 1] - 2 * cov[0, 1]
    p = float(2 * stats.norm.sf(abs(aucs[0] - aucs[1]) / np.sqrt(var))) if var > 0 else float("nan")
    return aucs[0], aucs[1], p


def mcnemar(c1, c2):
    """Exact McNemar test on two boolean correct-vectors."""
    b, c = int(np.sum(c1 & ~c2)), int(np.sum(~c1 & c2))
    return 1.0 if b + c == 0 else float(stats.binomtest(b, b + c, 0.5).pvalue)


def metrics(y, p):
    pred = p >= 0.5
    return {"acc": np.mean(pred == y), "auc": delong(p, p, y)[0],
            "sens": np.mean(pred[y == 1]), "spec": np.mean(~pred[y == 0])}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", required=True)
    ap.add_argument("--runs", nargs="*", default=[])
    ap.add_argument("--decision", nargs="*", default=[], help="single-modality runs to average")
    a = ap.parse_args()

    names = [a.ref] + a.runs + a.decision
    frames = {n: pd.read_csv(OOF_DIR / f"{n}.csv").set_index("scan_id") for n in names}
    common = sorted(set.intersection(*(set(d.index) for d in frames.values())))
    y = frames[a.ref].loc[common, "label"].to_numpy()
    probs = {n: frames[n].loc[common, "p_ad"].to_numpy() for n in [a.ref] + a.runs}
    if a.decision:
        probs["decision(" + "+".join(a.decision) + ")"] = np.mean(
            [frames[n].loc[common, "p_ad"].to_numpy() for n in a.decision], axis=0)

    print(f"{len(common)} paired scans (AD {int(y.sum())}, CN {len(y) - int(y.sum())})\n")
    print(f"{'run':<45}{'acc':>7}{'auc':>7}{'sens':>7}{'spec':>7}{'p_DeLong':>10}{'p_McNemar':>11}")
    ref_ok = (probs[a.ref] >= 0.5) == y
    for n, p in probs.items():
        m = metrics(y, p)
        if n == a.ref:
            pd_, pm = "", ""
        else:
            pd_ = f"{delong(p, probs[a.ref], y)[2]:.4f}"
            pm = f"{mcnemar((p >= 0.5) == y, ref_ok):.4f}"
        print(f"{n:<45}{m['acc']:7.3f}{m['auc']:7.3f}{m['sens']:7.3f}{m['spec']:7.3f}{pd_:>10}{pm:>11}")


if __name__ == "__main__":
    main()
