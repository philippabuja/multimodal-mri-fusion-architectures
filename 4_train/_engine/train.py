"""Train one model on one train/validation split.

    loss = CE + rg_weight * RG               (+ aux_weight * sum_m CE_m for --arch dyn)

The best epoch by validation cross-entropy is saved as best_model.pth next to config.json.
If training collapses (train CE stuck at ln 2) or a fusion branch dies, the run restarts with a
new seed (up to --max-attempts).

    python 4_train/_engine/train.py --modalities FLAIR --norm flair_lit \
        --split splits/fold_0.json --run-name FLAIR_cv/fold0 --data-root downsampled
"""

import argparse
import json
import math
import random
import re
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config import MODEL_DIR, weights_dir, data_root
sys.path.insert(0, str(MODEL_DIR))
from model import build_model
from dataset import NiftiDataset, NORM_FNS, resolve_norm_fns
from losses import RelevanceGuidedLoss
from constraints import BiasNegConstraint
from callbacks import ReduceLROnPlateau, EarlyStopping

BATCH_SIZE = 7
LR = 1e-3
COLLAPSE_CE_MIN, COLLAPSE_ACC_MAX, COLLAPSE_PATIENCE, COLLAPSE_GRACE = 0.690, 0.55, 3, 6

p = argparse.ArgumentParser()
p.add_argument("--modalities", nargs="+", default=["T1"])
p.add_argument("--norm", nargs="+", default=["none"], choices=sorted(NORM_FNS))
p.add_argument("--split", type=Path, required=True)
p.add_argument("--run-name", required=True)
p.add_argument("--data-root", choices=["full", "downsampled"], default="downsampled")
p.add_argument("--epochs", type=int, default=40)
p.add_argument("--early-stopping-patience", type=int, default=12)
p.add_argument("--seed", type=int, default=42)
p.add_argument("--max-attempts", type=int, default=3)
p.add_argument("--arch", choices=["early", "late", "dyn"], default="early")
p.add_argument("--width", type=int, default=8)
p.add_argument("--rg-weight", type=float, default=1.0, help="0 = RG-off ablation (DTD + bias<=0 stay on)")
p.add_argument("--rg-mode", choices=["pooled_sum", "per_modality"], default="pooled_sum")
p.add_argument("--rg-erosion", type=int, default=0, help="brain-mask erosion in voxels for the RG loss")
p.add_argument("--aux-weight", type=float, default=0.3, help="[dyn] weight of the per-modality CE")
p.add_argument("--weighting", choices=["confidence", "uniform"], default="confidence")
p.add_argument("--modality-dropout", type=float, default=0.0)
p.add_argument("--dropout-mode", choices=["auto", "independent", "solo_scaled"], default="auto")
p.add_argument("--warm-start-from", default=None,
               help="[dyn] run-prefix template of the single-modality runs, e.g. '{mod}_s{seed}_cv'; "
                    "encoder m is loaded from the SAME fold")
p.add_argument("--freeze-encoder-epochs", type=int, default=0)
p.add_argument("--encoder-lr-scale", type=float, default=1.0)
p.add_argument("--dead-branch-eps", type=float, default=1e-6)
args = p.parse_args()

MODS = args.modalities
M = len(MODS)
NORM = args.norm[0] if len(args.norm) == 1 else args.norm
N_BLOCKS, SHAPE = (4, (88, 112, 128)) if args.data_root == "downsampled" else (5, (176, 224, 256))
IS_DYN = args.arch == "dyn"
HAS_BRANCHES = args.arch in ("late", "dyn") and M > 1
_m = re.search(r"fold_(\d+)", args.split.stem)
FOLD = int(_m.group(1)) if _m else None
OUT = weights_dir(args.run_name)
OUT.mkdir(parents=True, exist_ok=True)
device = "cuda" if torch.cuda.is_available() else "cpu"


def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def scan_dirs(split):
    root = data_root(args.data_root)
    return [root / s for s in json.loads(args.split.read_text())[split]]


norm_fns = resolve_norm_fns(NORM, MODS)
train_set = NiftiDataset(scan_dirs("training"), MODS, norm_fns, SHAPE, args.rg_erosion)
val_set = NiftiDataset(scan_dirs("validation"), MODS, norm_fns, SHAPE, args.rg_erosion)
print(f"{args.run_name}: {MODS} arch={args.arch}  train {len(train_set)}  val {len(val_set)}")


def init_weights(m):
    if isinstance(m, (torch.nn.Linear, torch.nn.Conv3d)):
        torch.nn.init.kaiming_uniform_(m.weight, a=math.sqrt(6), mode="fan_in", nonlinearity="relu")
        torch.nn.init.zeros_(m.bias)


def run_training(seed):
    set_seed(seed)
    g = torch.Generator(); g.manual_seed(seed)
    pin = torch.cuda.is_available()
    train_loader = torch.utils.data.DataLoader(train_set, BATCH_SIZE, shuffle=True, num_workers=4,
                                               pin_memory=pin, generator=g)
    val_loader = torch.utils.data.DataLoader(val_set, BATCH_SIZE, shuffle=False, num_workers=4,
                                             pin_memory=pin)

    kw = dict(weighting=args.weighting, modality_dropout=args.modality_dropout,
              dropout_mode=args.dropout_mode) if IS_DYN else {}
    model = build_model(args.arch, in_channels=M, n_blocks=N_BLOCKS, with_dtd=True,
                        width=args.width, **kw).to(device)
    model.apply(init_weights)
    constraint = BiasNegConstraint()

    warm = []
    if IS_DYN and args.warm_start_from:
        ckpts = [weights_dir(f"{args.warm_start_from.format(mod=m, seed=args.seed)}/fold{FOLD}")
                 / "best_model.pth" for m in MODS]
        missing = [str(c) for c in ckpts if not c.exists()]
        if missing:
            sys.exit(f"warm start: missing checkpoints {missing}")
        warm = model.warm_start_encoders(ckpts)
        model.apply(constraint)

    if warm and args.encoder_lr_scale != 1.0:
        enc = {id(q) for q in model.encoders.parameters()}
        optimizer = torch.optim.Adam([
            {"params": list(model.encoders.parameters()), "lr": LR * args.encoder_lr_scale},
            {"params": [q for q in model.parameters() if id(q) not in enc], "lr": LR}])
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    ce_fn = torch.nn.CrossEntropyLoss()
    rg_fn = RelevanceGuidedLoss(mode=args.rg_mode)
    plateau = ReduceLROnPlateau()
    stopper = EarlyStopping(args.early_stopping_patience) if args.early_stopping_patience > 0 else None

    def run_epoch(loader, train):
        model.train(train)
        tot = {"loss": 0.0, "ce": 0.0, "rg": 0.0}
        correct, probs, labels_all, peak = 0, [], [], None
        with torch.set_grad_enabled(train):
            for x, mask, y in loader:
                x, mask, y = x.to(device), mask.to(device), y.to(device)
                logits, heat = model(x)
                ce, rg = ce_fn(logits, y), rg_fn(heat, mask)
                loss = ce + args.rg_weight * rg if args.rg_weight > 0 else ce
                if IS_DYN:
                    loss = loss + model.aux_loss(y, args.aux_weight)
                if HAS_BRANCHES:
                    act = model.bottleneck_activity.detach().max(dim=0).values
                    peak = act if peak is None else torch.maximum(peak, act)
                if train:
                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()
                    model.apply(constraint)
                tot["loss"] += loss.item(); tot["ce"] += ce.item(); tot["rg"] += rg.item()
                correct += (logits.argmax(1) == y).sum().item()
                probs.append(torch.softmax(logits.detach(), 1)[:, 1].cpu())
                labels_all.append(y.cpu())
        n = len(loader)
        pr, lb = torch.cat(probs).numpy(), torch.cat(labels_all).numpy()
        auc = float(roc_auc_score(lb, pr)) if len(set(lb)) > 1 else float("nan")
        return ({k: v / n for k, v in tot.items()} | {"acc": correct / len(loader.dataset), "auc": auc}), peak

    hist = {k: [] for k in ("t_loss", "v_loss", "t_ce", "v_ce", "t_rg", "v_rg", "t_acc", "v_acc", "v_auc")}
    best = {"ce": float("inf"), "epoch": None, "acc": None, "auc": None, "rg": None}
    collapsed, streak = False, 0

    for epoch in range(args.epochs):
        if warm and args.freeze_encoder_epochs > 0:
            for q in model.encoders.parameters():
                q.requires_grad_(epoch >= args.freeze_encoder_epochs)

        t, peak = run_epoch(train_loader, True)
        v, _ = run_epoch(val_loader, False)
        for k in ("loss", "ce", "rg", "acc"):
            hist[f"t_{k}"].append(t[k]); hist[f"v_{k}"].append(v[k])
        hist["v_auc"].append(v["auc"])
        print(f"ep {epoch + 1:2d}  train ce={t['ce']:.4f} rg={t['rg']:.4f} acc={t['acc']:.3f}  "
              f"val ce={v['ce']:.4f} rg={v['rg']:.4f} acc={v['acc']:.3f} auc={v['auc']:.3f}")

        healthy = v["ce"] < COLLAPSE_CE_MIN
        if healthy and v["ce"] < best["ce"]:
            best = {"ce": v["ce"], "epoch": epoch + 1, "acc": v["acc"], "auc": v["auc"], "rg": v["rg"]}
            torch.save(model.state_dict(), OUT / "best_model.pth")

        # a branch with an all-zero bottleneck gets zero gradient and never recovers
        if HAS_BRANCHES and peak is not None:
            dead = [MODS[i] for i in range(M) if float(peak[i]) < args.dead_branch_eps]
            if dead:
                print(f"[DEAD BRANCH] {dead} — restarting"); collapsed = True; break

        if epoch + 1 >= COLLAPSE_GRACE and t["ce"] >= COLLAPSE_CE_MIN and t["acc"] <= COLLAPSE_ACC_MAX:
            streak += 1
            if streak >= COLLAPSE_PATIENCE:
                print("[COLLAPSE] train CE at ln(2) — restarting"); collapsed = True; break
        else:
            streak = 0

        plateau.on_epoch_end(optimizer, v["loss"], is_healthy=healthy)
        if stopper and stopper.on_epoch_end(v["loss"]):
            break

    if best["epoch"] is None:
        collapsed = True
    return {"history": hist, "best": best, "collapsed": collapsed, "seed": seed}


for attempt in range(max(1, args.max_attempts)):
    result = run_training(args.seed + attempt)
    if not result["collapsed"]:
        break

b = result["best"]
(OUT / "config.json").write_text(json.dumps({
    "run_name": args.run_name, "split": str(args.split), "data_root": args.data_root,
    "modalities": MODS, "in_channels": M, "norm": NORM, "arch": args.arch, "width": args.width,
    "epochs": args.epochs, "seed": result["seed"], "attempts": attempt + 1,
    "collapsed": result["collapsed"],
    "rg_weight": args.rg_weight, "rg_mode": args.rg_mode, "rg_erosion": args.rg_erosion,
    "aux_weight": args.aux_weight if IS_DYN else None,
    "weighting": args.weighting if IS_DYN else None,
    "modality_dropout": args.modality_dropout if IS_DYN else None,
    "dropout_mode": args.dropout_mode if IS_DYN else None,
    "warm_start_from": args.warm_start_from,
    "best_epoch": b["epoch"], "best_val_ce": b["ce"] if b["epoch"] else None,
    "best_val_acc": b["acc"], "best_val_auc": b["auc"],
    "best_val_vsrim": -b["rg"] if b["rg"] is not None else None,   # relevance inside the brain
    "history": result["history"],
}, indent=2))
print("done.")
