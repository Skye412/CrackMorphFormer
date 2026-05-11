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
DEFAULT_BASE_DIR = "results/CrackMorphFormer_NewModel_RealOnly_Ablation"


# ============================================================
# Default training settings
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

POS_CROP_PROB = 0.75
DILATION_KERNEL = 21

DS_WEIGHTS = [0.5, 0.7, 1.0]
THRESHOLD = 0.5

CLI0U_TAUS = (2, 4, 6, 8)


# ============================================================
# Experiment variants
# ============================================================
VARIANTS = {
    "full_realonly": {
        "name": "full_realonly",
        "training": "real-only",
        "use_wds": True,
        "use_mpp": True,
        "do_pretrain": False,
        "description": "Full CrackMorphFormer trained only on S2DS.",
    },
    "full_synth2real": {
        "name": "full_synth2real",
        "training": "synth-to-real",
        "use_wds": True,
        "use_mpp": True,
        "do_pretrain": True,
        "description": "Full CrackMorphFormer with synthetic pretraining and S2DS fine-tuning.",
    },
    "wo_mpp_synth2real": {
        "name": "wo_mpp_synth2real",
        "training": "synth-to-real",
        "use_wds": True,
        "use_mpp": False,
        "do_pretrain": True,
        "description": "Ablation without MPP, with synthetic pretraining and S2DS fine-tuning.",
    },
    "wo_wds_synth2real": {
        "name": "wo_wds_synth2real",
        "training": "synth-to-real",
        "use_wds": False,
        "use_mpp": True,
        "do_pretrain": True,
        "description": "Ablation without WDS, with synthetic pretraining and S2DS fine-tuning.",
    },
}


# ============================================================
# Utilities
# ============================================================
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dirs(base_dir: str, variants: List[str]):
    os.makedirs(base_dir, exist_ok=True)
    os.makedirs(os.path.join(base_dir, "summary"), exist_ok=True)

    for key in variants:
        cfg = VARIANTS[key]
        variant_dir = os.path.join(base_dir, cfg["name"])

        os.makedirs(os.path.join(variant_dir, "weights"), exist_ok=True)
        os.makedirs(os.path.join(variant_dir, "txt"), exist_ok=True)

        if cfg["do_pretrain"]:
            os.makedirs(os.path.join(variant_dir, "pretrain", "weights"), exist_ok=True)
            os.makedirs(os.path.join(variant_dir, "pretrain", "txt"), exist_ok=True)


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
# Loss: BCE + Dice
# ============================================================
def compute_loss(pred_logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    BCE + Dice loss.

    Compatible labels:
        0   -> background
        1   -> crack
        255 -> ignored pixels, if present
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
        logits = output[-1] if isinstance(output, (list, tuple)) else output

        preds = (torch.sigmoid(logits) > threshold).float()

        valid_mask = (gts != 255)
        if not valid_mask.any():
            continue

        p_v = preds[valid_mask]
        g_v = gts[valid_mask]

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
# Builders
# ============================================================
def make_model(cfg: Dict, args):
    model = MyModel(
        channel=64,
        num_queries=args.num_queries,
        backbone_path=args.backbone_path,
        use_wds=cfg["use_wds"],
        use_mpp=cfg["use_mpp"],
    )
    return model.cuda()


def build_synth_loaders(args):
    train_loader = get_loader(
        args.synth_root,
        "synthcrack",
        "train",
        1,
        batchsize=args.batch_size,
        trainsize=args.train_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        pos_crop_prob=args.pos_crop_prob,
        dilation_kernel=args.dilation_kernel,
    )

    val_loader = get_loader(
        args.synth_root,
        "synthcrack",
        "val",
        1,
        batchsize=1,
        trainsize=args.train_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        pos_crop_prob=args.pos_crop_prob,
        dilation_kernel=args.dilation_kernel,
    )

    return train_loader, val_loader


def build_s2ds_loaders(args, fold: int):
    train_loader = get_loader(
        args.s2ds_root,
        "s2ds5",
        "train",
        fold,
        batchsize=args.batch_size,
        trainsize=args.train_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        pos_crop_prob=args.pos_crop_prob,
        dilation_kernel=args.dilation_kernel,
    )

    val_loader = get_loader(
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

    return train_loader, val_loader


# ============================================================
# Training
# ============================================================
def load_existing_and_eval(
    ckpt_path: str,
    val_loader,
    logger,
    stage_name: str,
    cfg: Dict,
    args,
) -> Dict[str, float]:
    model = make_model(cfg, args)

    state = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(state, strict=True)

    metrics = eval_metrics(
        val_loader=val_loader,
        model=model,
        threshold=args.threshold,
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
    cfg: Dict,
    args,
    load_from: Optional[str] = None,
    skip_existing: bool = False,
):
    if skip_existing and os.path.isfile(save_path):
        best_metrics = load_existing_and_eval(
            ckpt_path=save_path,
            val_loader=val_loader,
            logger=logger,
            stage_name=stage_name,
            cfg=cfg,
            args=args,
        )
        return save_path, best_metrics

    model = make_model(cfg, args)

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
        eta_min=args.eta_min,
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
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.grad_clip)
            optimizer.step()

            epoch_loss += loss.item()

        scheduler.step()

        metrics = eval_metrics(
            val_loader=val_loader,
            model=model,
            threshold=args.threshold,
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
# Variant flow
# ============================================================
def run_pretrain_for_variant(args, cfg: Dict, variant_dir: str):
    pretrain_dir = os.path.join(variant_dir, "pretrain")
    weights_dir = os.path.join(pretrain_dir, "weights")
    log_dir = os.path.join(pretrain_dir, "txt")

    os.makedirs(weights_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    ckpt_path = os.path.join(weights_dir, "best_pretrain.pth")
    log_file = os.path.join(log_dir, "pretrain_log.txt")

    logger = setup_logger(f"{cfg['name']}_pretrain", log_file)

    logger.info("=" * 100)
    logger.info(f"Stage 1: synthetic pretraining for {cfg['name']}")
    logger.info("=" * 100)
    logger.info(cfg["description"])
    logger.info(f"use_wds = {cfg['use_wds']}")
    logger.info(f"use_mpp = {cfg['use_mpp']}")
    logger.info(f"save_path = {ckpt_path}")
    logger.info(f"epochs = {args.synth_epochs}")
    logger.info(f"lr = {args.synth_lr}")
    logger.info(f"weight_decay = {args.synth_wd}")

    if args.reuse_pretrain and os.path.isfile(ckpt_path):
        logger.info(f"Reuse existing pretrain checkpoint: {ckpt_path}")
        return ckpt_path

    train_loader, val_loader = build_synth_loaders(args)

    ckpt, _ = train_one_stage(
        stage_name=f"{cfg['name']}_synth_pretrain",
        train_loader=train_loader,
        val_loader=val_loader,
        logger=logger,
        save_path=ckpt_path,
        epoch_num=args.synth_epochs,
        lr=args.synth_lr,
        weight_decay=args.synth_wd,
        cfg=cfg,
        args=args,
        load_from=None,
        skip_existing=False,
    )

    return ckpt


def run_variant(args, variant_key: str):
    cfg = VARIANTS[variant_key]
    variant_dir = os.path.join(args.base_dir, cfg["name"])

    weights_dir = os.path.join(variant_dir, "weights")
    log_dir = os.path.join(variant_dir, "txt")

    os.makedirs(weights_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    log_file = os.path.join(log_dir, "finetune_log.txt")
    logger = setup_logger(cfg["name"], log_file)

    logger.info("=" * 100)
    logger.info(f"Run variant: {cfg['name']}")
    logger.info("=" * 100)
    logger.info(cfg["description"])
    logger.info(f"training = {cfg['training']}")
    logger.info(f"use_wds = {cfg['use_wds']}")
    logger.info(f"use_mpp = {cfg['use_mpp']}")
    logger.info(f"do_pretrain = {cfg['do_pretrain']}")
    logger.info(f"folds = {args.folds}")
    logger.info(f"threshold = {args.threshold}")

    if cfg["do_pretrain"]:
        pretrain_ckpt = run_pretrain_for_variant(args, cfg, variant_dir)
    else:
        pretrain_ckpt = None

    rows = []

    for fold in args.folds:
        logger.info("\n" + "-" * 100)
        logger.info(f"Fine-tuning {cfg['name']} | fold {fold}")
        logger.info("-" * 100)

        train_loader, val_loader = build_s2ds_loaders(args, fold)

        save_path = os.path.join(weights_dir, f"best_finetune_fold{fold}.pth")

        _, best_metrics = train_one_stage(
            stage_name=f"{cfg['name']}_fold{fold}",
            train_loader=train_loader,
            val_loader=val_loader,
            logger=logger,
            save_path=save_path,
            epoch_num=args.finetune_epochs,
            lr=args.finetune_lr,
            weight_decay=args.finetune_wd,
            cfg=cfg,
            args=args,
            load_from=pretrain_ckpt,
            skip_existing=args.skip_existing,
        )

        row = {
            "variant": cfg["name"],
            "training": cfg["training"],
            "use_wds": cfg["use_wds"],
            "use_mpp": cfg["use_mpp"],
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


# ============================================================
# Summary
# ============================================================
def write_summary(all_rows: List[Dict], summary_csv: str):
    fieldnames = [
        "variant",
        "training",
        "use_wds",
        "use_mpp",
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

        for row in all_rows:
            writer.writerow({
                "variant": row["variant"],
                "training": row["training"],
                "use_wds": row["use_wds"],
                "use_mpp": row["use_mpp"],
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
    print("Final Summary")
    print("=" * 100)

    grouped = {}
    for row in all_rows:
        grouped.setdefault(row["variant"], []).append(row)

    for variant, rows in grouped.items():
        p = np.array([r["precision"] for r in rows], dtype=np.float64)
        rec = np.array([r["recall"] for r in rows], dtype=np.float64)
        f1 = np.array([r["f1"] for r in rows], dtype=np.float64)
        iou = np.array([r["iou"] for r in rows], dtype=np.float64)
        c2 = np.array([r["clIoU@2"] for r in rows], dtype=np.float64)
        c4 = np.array([r["clIoU@4"] for r in rows], dtype=np.float64)
        c6 = np.array([r["clIoU@6"] for r in rows], dtype=np.float64)
        c8 = np.array([r["clIoU@8"] for r in rows], dtype=np.float64)

        print(
            f"{variant}: "
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


# ============================================================
# CLI
# ============================================================
def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--synth-root", type=str, default=DEFAULT_SYNTH_ROOT)
    parser.add_argument("--s2ds-root", type=str, default=DEFAULT_S2DS_ROOT)
    parser.add_argument("--base-dir", type=str, default=DEFAULT_BASE_DIR)

    parser.add_argument(
        "--variants",
        nargs="+",
        default=[
            "full_realonly",
            "wo_mpp_synth2real",
            "wo_wds_synth2real",
        ],
        choices=list(VARIANTS.keys()),
        help="Variants to run.",
    )

    parser.add_argument(
        "--folds",
        nargs="+",
        type=int,
        default=[1, 2, 3, 4, 5],
    )

    parser.add_argument("--train-size", type=int, default=TRAIN_SIZE)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--num-queries", type=int, default=16)

    parser.add_argument("--synth-epochs", type=int, default=SYNTH_EPOCHS)
    parser.add_argument("--finetune-epochs", type=int, default=FINETUNE_EPOCHS)

    parser.add_argument("--synth-lr", type=float, default=SYNTH_LR)
    parser.add_argument("--finetune-lr", type=float, default=FINETUNE_LR)

    parser.add_argument("--synth-wd", type=float, default=SYNTH_WD)
    parser.add_argument("--finetune-wd", type=float, default=FINETUNE_WD)

    parser.add_argument("--eta-min", type=float, default=ETA_MIN)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--threshold", type=float, default=THRESHOLD)

    parser.add_argument("--pos-crop-prob", type=float, default=POS_CROP_PROB)
    parser.add_argument("--dilation-kernel", type=int, default=DILATION_KERNEL)

    parser.add_argument(
        "--backbone-path",
        type=str,
        default=None,
        help="Optional PVTv2-B2 pretrained weight path.",
    )

    parser.add_argument(
        "--reuse-pretrain",
        action="store_true",
        help="Reuse existing pretrain checkpoint if it already exists.",
    )

    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip a fold if its checkpoint already exists, but still evaluate it.",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    set_seed(SEED)
    torch.backends.cudnn.benchmark = True

    ensure_dirs(args.base_dir, args.variants)

    print("=" * 100)
    print("CrackMorphFormer new-model real-only and ablation runner")
    print("=" * 100)
    print(f"base_dir       = {args.base_dir}")
    print(f"synth_root     = {args.synth_root}")
    print(f"s2ds_root      = {args.s2ds_root}")
    print(f"variants       = {args.variants}")
    print(f"folds          = {args.folds}")
    print(f"train_size     = {args.train_size}")
    print(f"batch_size     = {args.batch_size}")
    print(f"num_queries    = {args.num_queries}")
    print(f"threshold      = {args.threshold}")
    print(f"reuse_pretrain = {args.reuse_pretrain}")
    print(f"skip_existing  = {args.skip_existing}")
    print(f"clIoU taus     = {CLI0U_TAUS}")

    all_rows = []

    for variant_key in args.variants:
        rows = run_variant(args, variant_key)
        all_rows.extend(rows)

    summary_csv = os.path.join(
        args.base_dir,
        "summary",
        "newmodel_realonly_ablation_summary.csv",
    )

    write_summary(all_rows, summary_csv)


if __name__ == "__main__":
    main()