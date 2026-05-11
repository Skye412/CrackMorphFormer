# -*- coding: utf-8 -*-
"""
Threshold sweep for CrackMorphFormer.

Usage example:

python tools/threshold_sweep.py \
  --base-dir results/CrackMorphFormer_final \
  --s2ds-root /home/skye/data/Skye/databases/s2ds5 \
  --experiments synth2real_full synth2real_wo_dfe synth2real_wo_sgmpp synth2real_wo_dfe_sgmpp \
  --folds 1 2 3 4 5 \
  --thresholds 0.30 0.35 0.40 0.45 0.50 0.55 0.60 0.65 0.70

Dense scan example:

python tools/threshold_sweep.py \
  --base-dir results/CrackMorphFormer_final \
  --s2ds-root /home/skye/data/Skye/databases/s2ds5 \
  --threshold-start 0.30 \
  --threshold-end 0.80 \
  --threshold-step 0.01
"""

import os
import csv
import argparse
from typing import Dict, List

import numpy as np
from tqdm import tqdm

import torch

from model.CrackMorphFormer import CrackMorphFormer
from ESDI_dataloader import get_loader


EXPERIMENTS = {
    "synth2real_full": {
        "use_dfe": True,
        "use_sgmpp": True,
    },
    "synth2real_wo_dfe": {
        "use_dfe": False,
        "use_sgmpp": True,
    },
    "synth2real_wo_sgmpp": {
        "use_dfe": True,
        "use_sgmpp": False,
    },
    "synth2real_wo_dfe_sgmpp": {
        "use_dfe": False,
        "use_sgmpp": False,
    },
}


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def load_model_state(model: torch.nn.Module, ckpt_path: str):
    state = torch.load(ckpt_path, map_location="cpu")

    if isinstance(state, dict) and "model_state" in state:
        state = state["model_state"]

    model.load_state_dict(state, strict=True)


def build_model(args, exp_name: str):
    cfg = EXPERIMENTS[exp_name]

    model = CrackMorphFormer(
        channel=args.channel,
        num_queries=args.num_queries,
        backbone_path=args.backbone_path,
        use_dfe=cfg["use_dfe"],
        use_sgmpp=cfg["use_sgmpp"],
    )

    return model.cuda().eval()


def build_val_loader(args, fold: int):
    return get_loader(
        args.s2ds_root,
        "s2ds5",
        "val",
        fold,
        batchsize=1,
        trainsize=args.train_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        pos_crop_prob=args.pos_crop_prob,
        dilation_kernel=args.dilation_kernel,
    )


def get_checkpoint_path(args, exp_name: str, fold: int):
    return os.path.join(
        args.base_dir,
        exp_name,
        "finetune",
        "weights",
        f"best_finetune_fold{fold}.pth",
    )


def make_thresholds(args):
    if args.thresholds is not None and len(args.thresholds) > 0:
        thresholds = [float(x) for x in args.thresholds]
    else:
        thresholds = np.arange(
            args.threshold_start,
            args.threshold_end + 1e-9,
            args.threshold_step,
            dtype=np.float64,
        ).tolist()

    thresholds = sorted(set([round(float(t), 6) for t in thresholds]))
    return thresholds


@torch.no_grad()
def sweep_one_checkpoint(
    model,
    val_loader,
    thresholds: List[float],
) -> Dict[float, Dict[str, float]]:
    """
    Fast threshold sweep.
    Computes P / R / F1 / IoU only.
    """

    stats = {
        t: {
            "tp": 0.0,
            "fp": 0.0,
            "fn": 0.0,
            "union": 0.0,
        }
        for t in thresholds
    }

    for data in tqdm(val_loader, desc="eval", leave=False):
        images = data["image"].cuda(non_blocking=True)
        gts = data["label"].cuda(non_blocking=True).float()

        outputs = model(images)

        if isinstance(outputs, (list, tuple)):
            logits = outputs[-1]
        else:
            logits = outputs

        probs = torch.sigmoid(logits)

        valid_mask = (gts != 255)

        if not valid_mask.any():
            continue

        g_valid = gts[valid_mask]

        for t in thresholds:
            preds = (probs > t).float()
            p_valid = preds[valid_mask]

            tp = (p_valid * g_valid).sum().item()
            fp = (p_valid * (1.0 - g_valid)).sum().item()
            fn = ((1.0 - p_valid) * g_valid).sum().item()
            union = ((p_valid + g_valid) > 0).sum().item()

            stats[t]["tp"] += tp
            stats[t]["fp"] += fp
            stats[t]["fn"] += fn
            stats[t]["union"] += union

    metrics = {}

    for t in thresholds:
        tp = stats[t]["tp"]
        fp = stats[t]["fp"]
        fn = stats[t]["fn"]
        union = stats[t]["union"]

        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2.0 * precision * recall / (precision + recall + 1e-8)
        iou = tp / (union + 1e-8)

        metrics[t] = {
            "threshold": t,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "iou": iou,
        }

    return metrics


def write_csv(rows: List[Dict], csv_path: str):
    ensure_dir(os.path.dirname(csv_path))

    if len(rows) == 0:
        return

    fieldnames = list(rows[0].keys())

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for r in rows:
            writer.writerow(r)


def summarize_best_by_fold(all_rows: List[Dict]) -> List[Dict]:
    grouped = {}

    for r in all_rows:
        key = (r["experiment"], r["fold"])
        grouped.setdefault(key, []).append(r)

    best_rows = []

    for (exp_name, fold), rows in grouped.items():
        rows_sorted = sorted(
            rows,
            key=lambda x: (
                float(x["f1"]),
                float(x["iou"]),
                float(x["clIoU@4"]) if "clIoU@4" in x else -1.0,
            ),
            reverse=True,
        )

        best = rows_sorted[0].copy()
        best["selection"] = "best_f1"
        best_rows.append(best)

    best_rows = sorted(
        best_rows,
        key=lambda x: (x["experiment"], int(x["fold"])),
    )

    return best_rows


def summarize_mean(best_rows: List[Dict]) -> List[Dict]:
    exp_names = sorted(set(r["experiment"] for r in best_rows))
    mean_rows = []

    for exp_name in exp_names:
        rs = [r for r in best_rows if r["experiment"] == exp_name]

        def arr(k):
            return np.array([float(r[k]) for r in rs], dtype=np.float64)

        threshold = arr("threshold")
        precision = arr("precision")
        recall = arr("recall")
        f1 = arr("f1")
        iou = arr("iou")

        row = {
            "experiment": exp_name,
            "folds": len(rs),
            "mean_best_threshold": f"{threshold.mean():.6f}",
            "precision_mean": f"{precision.mean():.6f}",
            "precision_std": f"{precision.std(ddof=1) if len(rs) > 1 else 0.0:.6f}",
            "recall_mean": f"{recall.mean():.6f}",
            "recall_std": f"{recall.std(ddof=1) if len(rs) > 1 else 0.0:.6f}",
            "f1_mean": f"{f1.mean():.6f}",
            "f1_std": f"{f1.std(ddof=1) if len(rs) > 1 else 0.0:.6f}",
            "iou_mean": f"{iou.mean():.6f}",
            "iou_std": f"{iou.std(ddof=1) if len(rs) > 1 else 0.0:.6f}",
        }

        mean_rows.append(row)

    return mean_rows


def print_mean_table(mean_rows: List[Dict]):
    print("\n" + "=" * 100)
    print("Best-threshold mean summary")
    print("=" * 100)

    for r in mean_rows:
        print(
            f"{r['experiment']}: "
            f"thr={r['mean_best_threshold']} | "
            f"P={r['precision_mean']}±{r['precision_std']} | "
            f"R={r['recall_mean']}±{r['recall_std']} | "
            f"F1={r['f1_mean']}±{r['f1_std']} | "
            f"IoU={r['iou_mean']}±{r['iou_std']}"
        )


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--base-dir",
        type=str,
        default="results/CrackMorphFormer_final",
    )

    parser.add_argument(
        "--s2ds-root",
        type=str,
        default="/home/skye/data/Skye/databases/s2ds5",
    )

    parser.add_argument(
        "--experiments",
        nargs="+",
        type=str,
        default=[
            "synth2real_full",
            "synth2real_wo_dfe",
            "synth2real_wo_sgmpp",
            "synth2real_wo_dfe_sgmpp",
        ],
        choices=list(EXPERIMENTS.keys()),
    )

    parser.add_argument(
        "--folds",
        nargs="+",
        type=int,
        default=[1, 2, 3, 4, 5],
    )

    parser.add_argument("--backbone-path", type=str, default=None)

    parser.add_argument("--channel", type=int, default=64)
    parser.add_argument("--num-queries", type=int, default=16)

    parser.add_argument("--train-size", type=int, default=512)
    parser.add_argument("--num-workers", type=int, default=4)

    parser.add_argument("--pos-crop-prob", type=float, default=0.75)
    parser.add_argument("--dilation-kernel", type=int, default=21)

    parser.add_argument(
        "--thresholds",
        nargs="+",
        type=float,
        default=None,
        help="Explicit threshold list. If set, overrides start/end/step.",
    )

    parser.add_argument("--threshold-start", type=float, default=0.30)
    parser.add_argument("--threshold-end", type=float, default=0.80)
    parser.add_argument("--threshold-step", type=float, default=0.01)

    parser.add_argument(
        "--out-dir",
        type=str,
        default=None,
    )

    return parser.parse_args()


def main():
    args = parse_args()

    thresholds = make_thresholds(args)

    if args.out_dir is None:
        args.out_dir = os.path.join(args.base_dir, "threshold_sweep")

    ensure_dir(args.out_dir)

    print("=" * 100)
    print("CrackMorphFormer threshold sweep")
    print("=" * 100)
    print(f"base_dir    = {args.base_dir}")
    print(f"s2ds_root   = {args.s2ds_root}")
    print(f"experiments = {args.experiments}")
    print(f"folds       = {args.folds}")
    print(f"thresholds  = {thresholds[0]:.3f} ... {thresholds[-1]:.3f}  n={len(thresholds)}")
    print(f"out_dir     = {args.out_dir}")
    print("=" * 100)

    all_rows = []

    for exp_name in args.experiments:
        for fold in args.folds:
            ckpt_path = get_checkpoint_path(args, exp_name, fold)

            print("\n" + "-" * 100)
            print(f"Experiment: {exp_name} | Fold: {fold}")
            print(f"Checkpoint: {ckpt_path}")
            print("-" * 100)

            if not os.path.isfile(ckpt_path):
                print(f"Missing checkpoint, skip: {ckpt_path}")
                continue

            val_loader = build_val_loader(args, fold)

            model = build_model(args, exp_name)
            load_model_state(model, ckpt_path)

            fold_metrics = sweep_one_checkpoint(
                model=model,
                val_loader=val_loader,
                thresholds=thresholds,
            )

            del model
            torch.cuda.empty_cache()

            for t in thresholds:
                m = fold_metrics[t]

                row = {
                    "experiment": exp_name,
                    "fold": fold,
                    "threshold": f"{m['threshold']:.6f}",
                    "precision": f"{m['precision']:.6f}",
                    "recall": f"{m['recall']:.6f}",
                    "f1": f"{m['f1']:.6f}",
                    "iou": f"{m['iou']:.6f}",
                    "checkpoint": ckpt_path,
                }

                all_rows.append(row)

            best_t = max(
                thresholds,
                key=lambda x: (
                    fold_metrics[x]["f1"],
                    fold_metrics[x]["iou"],
                ),
            )

            best = fold_metrics[best_t]

            print(
                f"Best threshold by F1: "
                f"thr={best_t:.3f} | "
                f"P={best['precision']:.6f} "
                f"R={best['recall']:.6f} "
                f"F1={best['f1']:.6f} "
                f"IoU={best['iou']:.6f}"
            )

    all_csv = os.path.join(args.out_dir, "threshold_sweep_all.csv")
    best_csv = os.path.join(args.out_dir, "threshold_sweep_best_by_fold.csv")
    mean_csv = os.path.join(args.out_dir, "threshold_sweep_best_mean.csv")

    write_csv(all_rows, all_csv)

    best_rows = summarize_best_by_fold(all_rows)
    write_csv(best_rows, best_csv)

    mean_rows = summarize_mean(best_rows)
    write_csv(mean_rows, mean_csv)

    print_mean_table(mean_rows)

    print("\nSaved:")
    print(all_csv)
    print(best_csv)
    print(mean_csv)


if __name__ == "__main__":
    main()