# -*- coding: utf-8 -*-
import os
import csv
import argparse
import random
import logging
from typing import Dict, List, Tuple, Optional

import numpy as np
from tqdm import tqdm

import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR

from skimage.morphology import binary_dilation, disk, thin

from model.CrackMorphFormer import CrackMorphFormer as MyModel
from ESDI_dataloader import get_loader


# ============================================================
# Default paths
# ============================================================
DEFAULT_SYNTH_ROOT = "/home/skye/data/Skye/databases/synthcrack"
DEFAULT_S2DS_ROOT = "/home/skye/data/Skye/databases/s2ds5"
DEFAULT_BASE_DIR = "results/CrackMorphFormer"


# ============================================================
# Training settings for A-mainline
# ============================================================
TRAIN_SIZE = 512
BATCH_SIZE = 4
SEED = 42

SYNTH_EPOCHS = 30
FINETUNE_EPOCHS = 60

SYNTH_LR = 8e-5
FINETUNE_LR = 5e-5

SYNTH_WD = 1e-4
FINETUNE_WD = 5e-4

ETA_MIN = 1e-7

# A-mainline crack-aware crop settings
POS_CROP_PROB = 0.75
DILATION_KERNEL = 21

# Keep the same deep-supervision weights used in the A experiments
DS_WEIGHTS = [0.5, 0.7, 1.0]

# Evaluation threshold
THRESHOLD = 0.5

# clIoU tolerance radii
CLI0U_TAUS = (2, 4, 6, 8)


# ============================================================
# Utils
# ============================================================
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dirs(base_dir: str):
    os.makedirs(base_dir, exist_ok=True)

    os.makedirs(os.path.join(base_dir, "shared_pretrain", "weights"), exist_ok=True)
    os.makedirs(os.path.join(base_dir, "shared_pretrain", "txt"), exist_ok=True)

    os.makedirs(os.path.join(base_dir, "A_bg_no_cldice", "weights"), exist_ok=True)
    os.makedirs(os.path.join(base_dir, "A_bg_no_cldice", "txt"), exist_ok=True)

    os.makedirs(os.path.join(base_dir, "summary"), exist_ok=True)


def setup_logger(name: str, log_file: str):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    if logger.handlers:
        logger.handlers.clear()

    formatter = logging.Formatter("%(message)s")

    fh = logging.FileHandler(log_file, mode="w", encoding="utf-8")
    fh.setFormatter(formatter)

    ch = logging.StreamHandler()
    ch.setFormatter(formatter)

    logger.addHandler(fh)
    logger.addHandler(ch)

    return logger


def default_metric_dict(value: float = 0.0) -> Dict[str, float]:
    return {
        "precision": value,
        "recall": value,
        "f1": value,
        "iou": value,
        "clIoU@2": value,
        "clIoU@4": value,
        "clIoU@6": value,
        "clIoU@8": value,
        "best_epoch": int(value),
    }


# ============================================================
# Loss: BCE + Dice only
# ============================================================
def compute_loss(pred_logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    BCE + Dice loss.

    The current A-mainline dataloader maps:
      crack -> 1
      background and other defects -> 0

    The valid-mask logic is retained for compatibility if 255 appears.
    """
    valid_mask = (target != 255).float()

    clean_target = target.clone()
    clean_target[target == 255] = 0.0

    bce = F.binary_cross_entropy_with_logits(
        pred_logits,
        clean_target,
        reduction="none",
    )
    bce = (bce * valid_mask).sum() / (valid_mask.sum() + 1e-8)

    pred = torch.sigmoid(pred_logits)
    inter = (pred * valid_mask * clean_target).sum()

    dice = 1.0 - (2.0 * inter + 1e-6) / (
        (pred * valid_mask).sum()
        + (clean_target * valid_mask).sum()
        + 1e-6
    )

    return bce + dice


# ============================================================
# Evaluation: Precision / Recall / F1 / IoU / clIoU
# ============================================================
def update_cl_stats(
    cl_stats: Dict[int, Dict[str, float]],
    pred_bin: np.ndarray,
    gt_bin: np.ndarray,
    taus: Tuple[int, ...],
):
    if pred_bin.sum() == 0 and gt_bin.sum() == 0:
        return

    skel_pred = thin(pred_bin)
    skel_gt = thin(gt_bin)

    for tau in taus:
        kernel = disk(tau)

        dil_pred = binary_dilation(skel_pred, footprint=kernel)
        dil_gt = binary_dilation(skel_gt, footprint=kernel)

        tp_cl = np.logical_and(skel_gt, dil_pred).sum()
        fp_cl = np.logical_and(skel_pred, np.logical_not(dil_gt)).sum()
        fn_cl = np.logical_and(skel_gt, np.logical_not(dil_pred)).sum()

        cl_stats[tau]["tp"] += float(tp_cl)
        cl_stats[tau]["fp"] += float(fp_cl)
        cl_stats[tau]["fn"] += float(fn_cl)


@torch.no_grad()
def eval_metrics(
    val_loader,
    model,
    threshold: float = THRESHOLD,
    taus: Tuple[int, ...] = CLI0U_TAUS,
) -> Dict[str, float]:
    model.eval()

    total_tp = 0.0
    total_fp = 0.0
    total_fn = 0.0
    total_union = 0.0

    cl_stats = {
        tau: {"tp": 0.0, "fp": 0.0, "fn": 0.0}
        for tau in taus
    }

    for data in val_loader:
        images = data["image"].cuda(non_blocking=True)
        gts = data["label"].cuda(non_blocking=True).float()

        output = model(images)
        if isinstance(output, (list, tuple)):
            logits = output[-1]
        else:
            logits = output

        preds = (torch.sigmoid(logits) > threshold).float()

        mask = (gts != 255)
        if not mask.any():
            continue

        p_v = preds[mask]
        g_v = gts[mask]

        total_tp += (p_v * g_v).sum().item()
        total_fp += (p_v * (1.0 - g_v)).sum().item()
        total_fn += ((1.0 - p_v) * g_v).sum().item()
        total_union += ((p_v + g_v) > 0).sum().item()

        preds_np = preds.detach().cpu().numpy()
        gts_np = gts.detach().cpu().numpy()

        for pred_i, gt_i in zip(preds_np, gts_np):
            pred_raw = pred_i[0].astype(bool)
            gt_raw = gt_i[0]

            valid = gt_raw != 255
            gt_bin = gt_raw == 1

            pred_bin = np.logical_and(pred_raw, valid)
            gt_bin = np.logical_and(gt_bin, valid)

            update_cl_stats(
                cl_stats=cl_stats,
                pred_bin=pred_bin,
                gt_bin=gt_bin,
                taus=taus,
            )

    precision = total_tp / (total_tp + total_fp + 1e-8)
    recall = total_tp / (total_tp + total_fn + 1e-8)
    f1 = 2.0 * precision * recall / (precision + recall + 1e-8)
    iou = total_tp / (total_union + 1e-8)

    metrics = {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "iou": iou,
    }

    for tau in taus:
        tp = cl_stats[tau]["tp"]
        fp = cl_stats[tau]["fp"]
        fn = cl_stats[tau]["fn"]
        metrics[f"clIoU@{tau}"] = tp / (tp + fp + fn + 1e-8)

    return metrics


# ============================================================
# Loader builders
# ============================================================
def build_synth_loaders(
    synth_root: str,
    batch_size: int,
    train_size: int,
    num_workers: int,
):
    train_loader = get_loader(
        synth_root,
        "synthcrack",
        "train",
        1,
        batchsize=batch_size,
        trainsize=train_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        pos_crop_prob=POS_CROP_PROB,
        dilation_kernel=DILATION_KERNEL,
    )

    val_loader = get_loader(
        synth_root,
        "synthcrack",
        "val",
        1,
        batchsize=1,
        trainsize=train_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        pos_crop_prob=POS_CROP_PROB,
        dilation_kernel=DILATION_KERNEL,
    )

    return train_loader, val_loader


def build_s2ds_loaders(
    s2ds_root: str,
    fold: int,
    batch_size: int,
    train_size: int,
    num_workers: int,
):
    train_loader = get_loader(
        s2ds_root,
        "s2ds5",
        "train",
        fold,
        batchsize=batch_size,
        trainsize=train_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        pos_crop_prob=POS_CROP_PROB,
        dilation_kernel=DILATION_KERNEL,
    )

    val_loader = get_loader(
        s2ds_root,
        "s2ds5",
        "val",
        fold,
        batchsize=1,
        trainsize=train_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        pos_crop_prob=POS_CROP_PROB,
        dilation_kernel=DILATION_KERNEL,
    )

    return train_loader, val_loader


# ============================================================
# Training helpers
# ============================================================
def load_existing_and_eval(
    ckpt_path: str,
    val_loader,
    logger,
    stage_name: str,
) -> Dict[str, float]:
    model = MyModel(channel=64).cuda()
    state = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(state, strict=True)

    metrics = eval_metrics(
        val_loader=val_loader,
        model=model,
        threshold=THRESHOLD,
        taus=CLI0U_TAUS,
    )
    metrics["best_epoch"] = -1

    logger.info(
        f"Skip training and evaluate existing checkpoint [{stage_name}] | "
        f"P:{metrics['precision']:.4f} "
        f"R:{metrics['recall']:.4f} "
        f"F1:{metrics['f1']:.4f} "
        f"IoU:{metrics['iou']:.4f} "
        f"clIoU@2:{metrics['clIoU@2']:.4f} "
        f"clIoU@4:{metrics['clIoU@4']:.4f} "
        f"clIoU@6:{metrics['clIoU@6']:.4f} "
        f"clIoU@8:{metrics['clIoU@8']:.4f}"
    )

    del model
    torch.cuda.empty_cache()

    return metrics


def train_one_stage(
    stage_name: str,
    train_loader,
    val_loader,
    logger,
    save_path: str,
    epoch_num: int,
    lr: float,
    weight_decay: float,
    load_from: Optional[str] = None,
    skip_existing: bool = False,
):
    if skip_existing and os.path.isfile(save_path):
        best_metrics = load_existing_and_eval(
            ckpt_path=save_path,
            val_loader=val_loader,
            logger=logger,
            stage_name=stage_name,
        )
        return save_path, best_metrics

    model = MyModel(channel=64).cuda()

    if load_from is not None:
        state = torch.load(load_from, map_location="cpu")
        model.load_state_dict(state, strict=True)
        logger.info(f"Loaded weights from: {load_from}")

    optimizer = optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )

    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=epoch_num,
        eta_min=ETA_MIN,
    )

    best_f1 = -1.0
    best_metrics = default_metric_dict(0.0)

    for ep in range(epoch_num):
        model.train()
        epoch_loss = 0.0

        pbar = tqdm(
            train_loader,
            desc=f"{stage_name} Ep {ep + 1}/{epoch_num}",
            leave=False,
        )

        for data in pbar:
            imgs = data["image"].cuda(non_blocking=True)
            gts = data["label"].cuda(non_blocking=True).float()

            optimizer.zero_grad(set_to_none=True)

            outputs = model(imgs)
            if not isinstance(outputs, (list, tuple)):
                outputs = [outputs]

            loss = 0.0
            for i, logit in enumerate(outputs):
                w = DS_WEIGHTS[i] if i < len(DS_WEIGHTS) else 1.0
                loss = loss + w * compute_loss(logit, gts)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss += loss.item()

        scheduler.step()

        metrics = eval_metrics(
            val_loader=val_loader,
            model=model,
            threshold=THRESHOLD,
            taus=CLI0U_TAUS,
        )

        precision = metrics["precision"]
        recall = metrics["recall"]
        f1 = metrics["f1"]
        iou = metrics["iou"]
        cliou2 = metrics["clIoU@2"]
        cliou4 = metrics["clIoU@4"]
        cliou6 = metrics["clIoU@6"]
        cliou8 = metrics["clIoU@8"]

        msg = (
            f"[{stage_name}] Ep {ep + 1}: "
            f"Loss:{epoch_loss / max(len(train_loader), 1):.4f} | "
            f"P:{precision:.4f} "
            f"R:{recall:.4f} "
            f"F1:{f1:.4f} "
            f"IoU:{iou:.4f} | "
            f"clIoU@2:{cliou2:.4f} "
            f"clIoU@4:{cliou4:.4f} "
            f"clIoU@6:{cliou6:.4f} "
            f"clIoU@8:{cliou8:.4f} | "
            f"LR:{optimizer.param_groups[0]['lr']:.8f}"
        )

        if f1 > best_f1:
            best_f1 = f1
            best_metrics = {
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "iou": iou,
                "clIoU@2": cliou2,
                "clIoU@4": cliou4,
                "clIoU@6": cliou6,
                "clIoU@8": cliou8,
                "best_epoch": ep + 1,
            }
            torch.save(model.state_dict(), save_path)
            msg += " Best"

        logger.info(msg)

    logger.info(
        f"Best [{stage_name}]: "
        f"Epoch={best_metrics['best_epoch']} | "
        f"P={best_metrics['precision']:.4f} "
        f"R={best_metrics['recall']:.4f} "
        f"F1={best_metrics['f1']:.4f} "
        f"IoU={best_metrics['iou']:.4f} "
        f"clIoU@2={best_metrics['clIoU@2']:.4f} "
        f"clIoU@4={best_metrics['clIoU@4']:.4f} "
        f"clIoU@6={best_metrics['clIoU@6']:.4f} "
        f"clIoU@8={best_metrics['clIoU@8']:.4f} | "
        f"ckpt={save_path}"
    )

    del model
    torch.cuda.empty_cache()

    return save_path, best_metrics


# ============================================================
# Main training flow
# ============================================================
def run_shared_pretrain(args, paths: Dict[str, str]):
    logger = setup_logger(
        "A_shared_pretrain",
        paths["shared_pretrain_log_file"],
    )

    logger.info("=" * 100)
    logger.info("Stage 1: Shared synthetic pretraining")
    logger.info("=" * 100)
    logger.info(f"SYNTH_ROOT = {args.synth_root}")
    logger.info(f"save_path = {paths['shared_pretrain_ckpt']}")
    logger.info(f"epochs = {args.synth_epochs}")
    logger.info(f"lr = {args.synth_lr}")
    logger.info(f"weight_decay = {args.synth_wd}")
    logger.info("loss = BCE + Dice")
    logger.info("clDice = disabled")

    if args.pretrain_weights is not None:
        if not os.path.isfile(args.pretrain_weights):
            raise FileNotFoundError(f"Provided pretrain checkpoint not found: {args.pretrain_weights}")
        logger.info(f"Use external pretrain checkpoint: {args.pretrain_weights}")
        return args.pretrain_weights

    if args.reuse_pretrain and os.path.isfile(paths["shared_pretrain_ckpt"]):
        logger.info(f"Reuse existing shared pretrain: {paths['shared_pretrain_ckpt']}")
        return paths["shared_pretrain_ckpt"]

    train_loader, val_loader = build_synth_loaders(
        synth_root=args.synth_root,
        batch_size=args.batch_size,
        train_size=args.train_size,
        num_workers=args.num_workers,
    )

    ckpt, _ = train_one_stage(
        stage_name="shared_synth_pretrain",
        train_loader=train_loader,
        val_loader=val_loader,
        logger=logger,
        save_path=paths["shared_pretrain_ckpt"],
        epoch_num=args.synth_epochs,
        lr=args.synth_lr,
        weight_decay=args.synth_wd,
        load_from=None,
        skip_existing=False,
    )

    return ckpt


def run_finetune_A(
    args,
    paths: Dict[str, str],
    pretrain_weights: str,
    folds: List[int],
):
    logger = setup_logger(
        "A_bg_no_cldice",
        paths["finetune_log_file"],
    )

    logger.info("=" * 100)
    logger.info("Stage 2: A-mainline fine-tuning")
    logger.info("=" * 100)
    logger.info("Variant = A_bg_no_cldice")
    logger.info("S2DS policy = other defect colors as background")
    logger.info("Loss = BCE + Dice only")
    logger.info("clDice = disabled")
    logger.info(f"pretrain_weights = {pretrain_weights}")
    logger.info(f"folds = {folds}")
    logger.info(f"epochs = {args.finetune_epochs}")
    logger.info(f"lr = {args.finetune_lr}")
    logger.info(f"weight_decay = {args.finetune_wd}")
    logger.info(f"threshold = {THRESHOLD}")
    logger.info(f"pos_crop_prob = {POS_CROP_PROB}")
    logger.info(f"dilation_kernel = {DILATION_KERNEL}")
    logger.info("post-processing = disabled")

    rows = []

    for fold in folds:
        logger.info("\n" + "-" * 100)
        logger.info(f"Fine-tuning A-mainline | fold {fold}")
        logger.info("-" * 100)

        train_loader, val_loader = build_s2ds_loaders(
            s2ds_root=args.s2ds_root,
            fold=fold,
            batch_size=args.batch_size,
            train_size=args.train_size,
            num_workers=args.num_workers,
        )

        save_path = os.path.join(
            paths["finetune_weights_dir"],
            f"best_finetune_fold{fold}.pth",
        )

        _, best_metrics = train_one_stage(
            stage_name=f"A_bg_no_cldice_fold{fold}",
            train_loader=train_loader,
            val_loader=val_loader,
            logger=logger,
            save_path=save_path,
            epoch_num=args.finetune_epochs,
            lr=args.finetune_lr,
            weight_decay=args.finetune_wd,
            load_from=pretrain_weights,
            skip_existing=args.skip_existing,
        )

        row = {
            "variant": "A_bg_no_cldice",
            "fold": fold,
            "best_epoch": best_metrics["best_epoch"],
            "precision": best_metrics["precision"],
            "recall": best_metrics["recall"],
            "f1": best_metrics["f1"],
            "iou": best_metrics["iou"],
            "clIoU@2": best_metrics["clIoU@2"],
            "clIoU@4": best_metrics["clIoU@4"],
            "clIoU@6": best_metrics["clIoU@6"],
            "clIoU@8": best_metrics["clIoU@8"],
            "checkpoint": save_path,
        }
        rows.append(row)

        logger.info(
            f"Fold {fold} done | "
            f"BestEpoch={best_metrics['best_epoch']} | "
            f"P={best_metrics['precision']:.4f} "
            f"R={best_metrics['recall']:.4f} "
            f"F1={best_metrics['f1']:.4f} "
            f"IoU={best_metrics['iou']:.4f} "
            f"clIoU@2={best_metrics['clIoU@2']:.4f} "
            f"clIoU@4={best_metrics['clIoU@4']:.4f} "
            f"clIoU@6={best_metrics['clIoU@6']:.4f} "
            f"clIoU@8={best_metrics['clIoU@8']:.4f}"
        )

    return rows


def write_summary(rows: List[Dict], summary_csv: str):
    fieldnames = [
        "variant",
        "fold",
        "best_epoch",
        "precision",
        "recall",
        "f1",
        "iou",
        "clIoU@2",
        "clIoU@4",
        "clIoU@6",
        "clIoU@8",
        "checkpoint",
    ]

    with open(summary_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for row in rows:
            writer.writerow({
                "variant": row["variant"],
                "fold": row["fold"],
                "best_epoch": row["best_epoch"],
                "precision": f"{row['precision']:.6f}",
                "recall": f"{row['recall']:.6f}",
                "f1": f"{row['f1']:.6f}",
                "iou": f"{row['iou']:.6f}",
                "clIoU@2": f"{row['clIoU@2']:.6f}",
                "clIoU@4": f"{row['clIoU@4']:.6f}",
                "clIoU@6": f"{row['clIoU@6']:.6f}",
                "clIoU@8": f"{row['clIoU@8']:.6f}",
                "checkpoint": row["checkpoint"],
            })

    print("\n" + "=" * 100)
    print("Final A-mainline Summary")
    print("=" * 100)

    p = np.array([r["precision"] for r in rows], dtype=np.float64)
    rec = np.array([r["recall"] for r in rows], dtype=np.float64)
    f1 = np.array([r["f1"] for r in rows], dtype=np.float64)
    iou = np.array([r["iou"] for r in rows], dtype=np.float64)
    c2 = np.array([r["clIoU@2"] for r in rows], dtype=np.float64)
    c4 = np.array([r["clIoU@4"] for r in rows], dtype=np.float64)
    c6 = np.array([r["clIoU@6"] for r in rows], dtype=np.float64)
    c8 = np.array([r["clIoU@8"] for r in rows], dtype=np.float64)

    print(
        f"A_bg_no_cldice: "
        f"P={p.mean():.4f}±{p.std():.4f} | "
        f"R={rec.mean():.4f}±{rec.std():.4f} | "
        f"F1={f1.mean():.4f}±{f1.std():.4f} | "
        f"IoU={iou.mean():.4f}±{iou.std():.4f} | "
        f"clIoU@2={c2.mean():.4f}±{c2.std():.4f} | "
        f"clIoU@4={c4.mean():.4f}±{c4.std():.4f} | "
        f"clIoU@6={c6.mean():.4f}±{c6.std():.4f} | "
        f"clIoU@8={c8.mean():.4f}±{c8.std():.4f}"
    )

    print(f"\nSaved summary: {summary_csv}")


def build_paths(base_dir: str) -> Dict[str, str]:
    shared_pretrain_dir = os.path.join(base_dir, "shared_pretrain")
    shared_pretrain_weights_dir = os.path.join(shared_pretrain_dir, "weights")
    shared_pretrain_log_dir = os.path.join(shared_pretrain_dir, "txt")

    finetune_dir = os.path.join(base_dir, "A_bg_no_cldice")
    finetune_weights_dir = os.path.join(finetune_dir, "weights")
    finetune_log_dir = os.path.join(finetune_dir, "txt")

    summary_dir = os.path.join(base_dir, "summary")

    return {
        "shared_pretrain_weights_dir": shared_pretrain_weights_dir,
        "shared_pretrain_log_dir": shared_pretrain_log_dir,
        "shared_pretrain_log_file": os.path.join(shared_pretrain_log_dir, "training_log.txt"),
        "shared_pretrain_ckpt": os.path.join(shared_pretrain_weights_dir, "best_pretrain_fold1.pth"),
        "finetune_dir": finetune_dir,
        "finetune_weights_dir": finetune_weights_dir,
        "finetune_log_dir": finetune_log_dir,
        "finetune_log_file": os.path.join(finetune_log_dir, "training_log.txt"),
        "summary_dir": summary_dir,
        "summary_csv": os.path.join(summary_dir, "A_mainline_summary.csv"),
    }


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--synth-root", type=str, default=DEFAULT_SYNTH_ROOT)
    parser.add_argument("--s2ds-root", type=str, default=DEFAULT_S2DS_ROOT)
    parser.add_argument("--base-dir", type=str, default=DEFAULT_BASE_DIR)

    parser.add_argument(
        "--folds",
        nargs="+",
        type=int,
        default=[1, 2, 3, 4, 5],
    )

    parser.add_argument("--train-size", type=int, default=TRAIN_SIZE)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--num-workers", type=int, default=4)

    parser.add_argument("--synth-epochs", type=int, default=SYNTH_EPOCHS)
    parser.add_argument("--finetune-epochs", type=int, default=FINETUNE_EPOCHS)

    parser.add_argument("--synth-lr", type=float, default=SYNTH_LR)
    parser.add_argument("--finetune-lr", type=float, default=FINETUNE_LR)

    parser.add_argument("--synth-wd", type=float, default=SYNTH_WD)
    parser.add_argument("--finetune-wd", type=float, default=FINETUNE_WD)

    parser.add_argument(
        "--reuse-pretrain",
        action="store_true",
        help="Reuse shared pretrain checkpoint if it already exists.",
    )

    parser.add_argument(
        "--pretrain-weights",
        type=str,
        default=None,
        help="Use an external pretrain checkpoint and skip Stage 1.",
    )

    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip a fine-tuning fold if its checkpoint already exists, but still evaluate it.",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    set_seed(SEED)
    torch.backends.cudnn.benchmark = True

    ensure_dirs(args.base_dir)
    paths = build_paths(args.base_dir)

    print("=" * 100)
    print("CrackMorphFormer-A mainline training")
    print("=" * 100)
    print(f"base_dir         = {args.base_dir}")
    print(f"synth_root       = {args.synth_root}")
    print(f"s2ds_root        = {args.s2ds_root}")
    print(f"folds            = {args.folds}")
    print(f"train_size       = {args.train_size}")
    print(f"batch_size       = {args.batch_size}")
    print(f"reuse_pretrain   = {args.reuse_pretrain}")
    print(f"pretrain_weights = {args.pretrain_weights}")
    print(f"skip_existing    = {args.skip_existing}")
    print(f"threshold        = {THRESHOLD}")
    print(f"clIoU taus       = {CLI0U_TAUS}")
    print("variant          = A_bg_no_cldice")
    print("loss             = BCE + Dice")
    print("clDice           = disabled")
    print("postprocess      = disabled")

    pretrain_weights = run_shared_pretrain(args, paths)

    rows = run_finetune_A(
        args=args,
        paths=paths,
        pretrain_weights=pretrain_weights,
        folds=args.folds,
    )

    write_summary(
        rows=rows,
        summary_csv=paths["summary_csv"],
    )


if __name__ == "__main__":
    main()