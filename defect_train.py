# -*- coding: utf-8 -*-
"""
Training script for CrackMorphFormer.

Default experiment:
    synth2real_full

Key change:
    During each epoch, validation only computes Precision / Recall / F1 / IoU.
    clIoU@2/4/6/8 is computed only once for the best-F1 checkpoint of each fold.

Supported experiments:
    real_only_full
    synth2real_full
    synth2real_wo_dfe
    synth2real_wo_sgmpp

Model requirement:
    model/CrackMorphFormer.py should define:

        class CrackMorphFormer(nn.Module):
            def __init__(
                self,
                channel=64,
                num_queries=16,
                backbone_path=None,
                use_dfe=True,
                use_sgmpp=True,
            )
"""

import os
import csv
import time
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
DEFAULT_BASE_DIR = "/home/skye/data/Skye/CrackMorphFormer/results/CrackMorphFormer_final"


# ============================================================
# Default training settings
# ============================================================

MODEL_NAME = "CrackMorphFormer"

SEED = 42

TRAIN_SIZE = 512
BATCH_SIZE = 4
NUM_WORKERS = 4

SYNTH_EPOCHS = 30
FINETUNE_EPOCHS = 60

SYNTH_LR = 8e-5
FINETUNE_LR = 5e-5

SYNTH_WD = 1e-4
FINETUNE_WD = 5e-4

ETA_MIN = 1e-7
GRAD_CLIP = 1.0

CHANNEL = 64
NUM_QUERIES = 16

POS_CROP_PROB = 0.75
DILATION_KERNEL = 21

THRESHOLD = 0.5

DS_WEIGHTS = [0.5, 0.7, 1.0]

CLI0U_TAUS = (2, 4, 6, 8)


EXPERIMENTS = {
    "synth2real_full": {
        "use_synth_pretrain": True,
        "use_dfe": True,
        "use_sgmpp": True,
    },
    "synth2real_wo_dfe": {
        "use_synth_pretrain": True,
        "use_dfe": False,
        "use_sgmpp": True,
    },
    "synth2real_wo_sgmpp": {
        "use_synth_pretrain": True,
        "use_dfe": True,
        "use_sgmpp": False,
    },
    "synth2real_wo_dfe_sgmpp": {
        "use_synth_pretrain": True,
        "use_dfe": False,
        "use_sgmpp": False,
    },
}


# ============================================================
# Utils
# ============================================================

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


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


def count_trainable_params(model: torch.nn.Module) -> float:
    return sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6


def safe_float(x):
    try:
        return float(x)
    except Exception:
        return 0.0


# ============================================================
# Loss
# ============================================================

def compute_loss(pred_logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    BCE + Dice loss.

    target:
        0: background
        1: crack
        255: ignored pixel, if present
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
# Metrics
# ============================================================

def update_cl_stats(
    cl_stats: Dict[int, Dict[str, float]],
    pred_bin: np.ndarray,
    gt_bin: np.ndarray,
    taus: Tuple[int, ...],
):
    """
    Update clIoU statistics for one binary mask pair.

    pred_bin: bool array [H, W]
    gt_bin:   bool array [H, W]
    """
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
    threshold: float = 0.5,
    compute_cliou: bool = False,
    taus: Tuple[int, ...] = CLI0U_TAUS,
) -> Dict[str, float]:
    """
    Evaluation.

    compute_cliou=False:
        fast validation used every epoch.

    compute_cliou=True:
        slow validation used only for the best checkpoint.
    """
    model.eval()

    total_tp = 0.0
    total_fp = 0.0
    total_fn = 0.0
    total_union = 0.0

    cl_stats = None

    if compute_cliou:
        cl_stats = {
            tau: {"tp": 0.0, "fp": 0.0, "fn": 0.0}
            for tau in taus
        }

    for data in val_loader:
        images = data["image"].cuda(non_blocking=True)
        gts = data["label"].cuda(non_blocking=True).float()

        outputs = model(images)

        if isinstance(outputs, (list, tuple)):
            logits = outputs[-1]
        else:
            logits = outputs

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

        if compute_cliou:
            preds_np = preds.detach().cpu().numpy()
            gts_np = gts.detach().cpu().numpy()

            for pred_i, gt_i in zip(preds_np, gts_np):
                pred_raw = pred_i[0].astype(bool)
                gt_raw = gt_i[0]

                valid = gt_raw != 255

                pred_bin = np.logical_and(pred_raw, valid)
                gt_bin = np.logical_and(gt_raw == 1, valid)

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

    if compute_cliou:
        for tau in taus:
            tp = cl_stats[tau]["tp"]
            fp = cl_stats[tau]["fp"]
            fn = cl_stats[tau]["fn"]

            metrics[f"clIoU@{tau}"] = tp / (tp + fp + fn + 1e-8)
    else:
        for tau in taus:
            metrics[f"clIoU@{tau}"] = -1.0

    return metrics


# ============================================================
# Data loaders
# ============================================================

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
# Checkpoint
# ============================================================

def save_model_state(model: torch.nn.Module, save_path: str):
    ensure_dir(os.path.dirname(save_path))
    torch.save(model.state_dict(), save_path)


def load_model_state(
    model: torch.nn.Module,
    ckpt_path: str,
    strict: bool = True,
):
    state = torch.load(ckpt_path, map_location="cpu")

    if isinstance(state, dict) and "model_state" in state:
        state = state["model_state"]

    return model.load_state_dict(state, strict=strict)


def build_model(args, use_dfe: bool, use_sgmpp: bool):
    model = MyModel(
        channel=args.channel,
        num_queries=args.num_queries,
        backbone_path=args.backbone_path,
        use_dfe=use_dfe,
        use_sgmpp=use_sgmpp,
    )
    return model.cuda()


# ============================================================
# Training
# ============================================================

def train_one_stage(
    stage_name: str,
    train_loader,
    val_loader,
    logger,
    save_path: str,
    epoch_num: int,
    lr: float,
    weight_decay: float,
    args,
    use_dfe: bool,
    use_sgmpp: bool,
    load_from: Optional[str] = None,
    skip_existing: bool = False,
):
    """
    Train one stage and save checkpoint by best F1.

    During training:
        validation computes P/R/F1/IoU only.

    After training:
        load best checkpoint and compute clIoU once.
    """
    if skip_existing and os.path.isfile(save_path):
        logger.info(f"Skip existing checkpoint: {save_path}")

        model = build_model(
            args=args,
            use_dfe=use_dfe,
            use_sgmpp=use_sgmpp,
        )

        load_model_state(
            model=model,
            ckpt_path=save_path,
            strict=True,
        )

        metrics = eval_metrics(
            val_loader=val_loader,
            model=model,
            threshold=args.threshold,
            compute_cliou=True,
            taus=CLI0U_TAUS,
        )

        metrics["best_epoch"] = -1

        del model
        torch.cuda.empty_cache()

        return save_path, metrics

    model = build_model(
        args=args,
        use_dfe=use_dfe,
        use_sgmpp=use_sgmpp,
    )

    logger.info(f"Model: {MODEL_NAME}")
    logger.info(f"Stage: {stage_name}")
    logger.info(f"use_dfe = {use_dfe}")
    logger.info(f"use_sgmpp = {use_sgmpp}")
    logger.info(f"Trainable params: {count_trainable_params(model):.2f}M")

    if load_from is not None:
        load_model_state(
            model=model,
            ckpt_path=load_from,
            strict=True,
        )
        logger.info(f"Loaded pretrained weights from: {load_from}")

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
    best_epoch = 0

    for ep in range(epoch_num):
        model.train()

        epoch_loss = 0.0
        tic = time.time()

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

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=args.grad_clip,
            )

            optimizer.step()

            epoch_loss += float(loss.item())

            pbar.set_postfix(
                loss=f"{loss.item():.4f}",
                lr=f"{optimizer.param_groups[0]['lr']:.2e}",
            )

        scheduler.step()

        # Fast validation. No clIoU here.
        metrics = eval_metrics(
            val_loader=val_loader,
            model=model,
            threshold=args.threshold,
            compute_cliou=False,
            taus=CLI0U_TAUS,
        )

        precision = metrics["precision"]
        recall = metrics["recall"]
        f1 = metrics["f1"]
        iou = metrics["iou"]

        epoch_time = time.time() - tic

        msg = (
            f"[{stage_name}] "
            f"Ep {ep + 1:03d}/{epoch_num} | "
            f"Loss:{epoch_loss / max(len(train_loader), 1):.4f} | "
            f"P:{precision:.4f} "
            f"R:{recall:.4f} "
            f"F1:{f1:.4f} "
            f"IoU:{iou:.4f} | "
            f"LR:{optimizer.param_groups[0]['lr']:.8f} | "
            f"Time:{epoch_time:.1f}s"
        )

        if f1 > best_f1:
            best_f1 = f1
            best_epoch = ep + 1

            save_model_state(
                model=model,
                save_path=save_path,
            )

            msg += "  Best"

        logger.info(msg)

    # Compute clIoU only once for best-F1 checkpoint.
    logger.info("-" * 100)
    logger.info(f"Compute clIoU for best checkpoint only: {save_path}")
    logger.info("-" * 100)

    load_model_state(
        model=model,
        ckpt_path=save_path,
        strict=True,
    )

    best_metrics = eval_metrics(
        val_loader=val_loader,
        model=model,
        threshold=args.threshold,
        compute_cliou=True,
        taus=CLI0U_TAUS,
    )

    best_metrics["best_epoch"] = best_epoch

    logger.info(
        f"Best [{stage_name}] | "
        f"Epoch:{best_epoch} | "
        f"P:{best_metrics['precision']:.6f} "
        f"R:{best_metrics['recall']:.6f} "
        f"F1:{best_metrics['f1']:.6f} "
        f"IoU:{best_metrics['iou']:.6f} "
        f"clIoU@2:{best_metrics['clIoU@2']:.6f} "
        f"clIoU@4:{best_metrics['clIoU@4']:.6f} "
        f"clIoU@6:{best_metrics['clIoU@6']:.6f} "
        f"clIoU@8:{best_metrics['clIoU@8']:.6f} | "
        f"ckpt:{save_path}"
    )

    del model
    torch.cuda.empty_cache()

    return save_path, best_metrics


# ============================================================
# Paths and CSV
# ============================================================

def build_exp_paths(base_dir: str, exp_name: str) -> Dict[str, str]:
    exp_dir = os.path.join(base_dir, exp_name)

    paths = {
        "exp_dir": exp_dir,

        "pretrain_dir": os.path.join(exp_dir, "pretrain"),
        "pretrain_weights_dir": os.path.join(exp_dir, "pretrain", "weights"),
        "pretrain_log_dir": os.path.join(exp_dir, "pretrain", "txt"),
        "pretrain_ckpt": os.path.join(exp_dir, "pretrain", "weights", "best_pretrain.pth"),
        "pretrain_log_file": os.path.join(exp_dir, "pretrain", "txt", "training_log.txt"),

        "finetune_dir": os.path.join(exp_dir, "finetune"),
        "finetune_weights_dir": os.path.join(exp_dir, "finetune", "weights"),
        "finetune_log_dir": os.path.join(exp_dir, "finetune", "txt"),
        "finetune_log_file": os.path.join(exp_dir, "finetune", "txt", "training_log.txt"),

        "summary_dir": os.path.join(exp_dir, "summary"),
        "summary_csv": os.path.join(exp_dir, "summary", f"{exp_name}_summary.csv"),
    }

    for k, v in paths.items():
        if k.endswith("_dir") or k in ["exp_dir", "summary_dir"]:
            ensure_dir(v)

    return paths


def write_rows_csv(rows: List[Dict], csv_path: str):
    ensure_dir(os.path.dirname(csv_path))

    fieldnames = [
        "experiment",
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
        "use_synth_pretrain",
        "use_dfe",
        "use_sgmpp",
        "checkpoint",
    ]

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for r in rows:
            writer.writerow({
                "experiment": r["experiment"],
                "fold": r["fold"],
                "best_epoch": r["best_epoch"],
                "precision": f"{r['precision']:.6f}",
                "recall": f"{r['recall']:.6f}",
                "f1": f"{r['f1']:.6f}",
                "iou": f"{r['iou']:.6f}",
                "clIoU@2": f"{r['clIoU@2']:.6f}",
                "clIoU@4": f"{r['clIoU@4']:.6f}",
                "clIoU@6": f"{r['clIoU@6']:.6f}",
                "clIoU@8": f"{r['clIoU@8']:.6f}",
                "use_synth_pretrain": r["use_synth_pretrain"],
                "use_dfe": r["use_dfe"],
                "use_sgmpp": r["use_sgmpp"],
                "checkpoint": r["checkpoint"],
            })


def print_summary(rows: List[Dict]):
    if len(rows) == 0:
        return

    print("\n" + "=" * 100)
    print("Current Summary")
    print("=" * 100)

    exp_names = sorted(set(r["experiment"] for r in rows))

    for exp_name in exp_names:
        rs = [r for r in rows if r["experiment"] == exp_name]

        p = np.array([r["precision"] for r in rs], dtype=np.float64)
        rec = np.array([r["recall"] for r in rs], dtype=np.float64)
        f1 = np.array([r["f1"] for r in rs], dtype=np.float64)
        iou = np.array([r["iou"] for r in rs], dtype=np.float64)
        c2 = np.array([r["clIoU@2"] for r in rs], dtype=np.float64)
        c4 = np.array([r["clIoU@4"] for r in rs], dtype=np.float64)
        c6 = np.array([r["clIoU@6"] for r in rs], dtype=np.float64)
        c8 = np.array([r["clIoU@8"] for r in rs], dtype=np.float64)

        print(
            f"{exp_name}: "
            f"P={p.mean():.4f}±{p.std(ddof=1) if len(p) > 1 else 0.0:.4f} | "
            f"R={rec.mean():.4f}±{rec.std(ddof=1) if len(rec) > 1 else 0.0:.4f} | "
            f"F1={f1.mean():.4f}±{f1.std(ddof=1) if len(f1) > 1 else 0.0:.4f} | "
            f"IoU={iou.mean():.4f}±{iou.std(ddof=1) if len(iou) > 1 else 0.0:.4f} | "
            f"clIoU@2={c2.mean():.4f}±{c2.std(ddof=1) if len(c2) > 1 else 0.0:.4f} | "
            f"clIoU@4={c4.mean():.4f}±{c4.std(ddof=1) if len(c4) > 1 else 0.0:.4f} | "
            f"clIoU@6={c6.mean():.4f}±{c6.std(ddof=1) if len(c6) > 1 else 0.0:.4f} | "
            f"clIoU@8={c8.mean():.4f}±{c8.std(ddof=1) if len(c8) > 1 else 0.0:.4f}"
        )


# ============================================================
# Experiment runners
# ============================================================

def run_pretrain(args, exp_name: str, exp_cfg: Dict, paths: Dict[str, str]) -> str:
    logger = setup_logger(
        name=f"{exp_name}_pretrain",
        log_file=paths["pretrain_log_file"],
    )

    logger.info("=" * 100)
    logger.info(f"Stage 1: Synthetic pretraining | {exp_name}")
    logger.info("=" * 100)
    logger.info(f"use_dfe = {exp_cfg['use_dfe']}")
    logger.info(f"use_sgmpp = {exp_cfg['use_sgmpp']}")
    logger.info(f"synth_root = {args.synth_root}")
    logger.info(f"epochs = {args.synth_epochs}")
    logger.info(f"lr = {args.synth_lr}")
    logger.info(f"weight_decay = {args.synth_wd}")
    logger.info("Epoch validation: P/R/F1/IoU only")
    logger.info("clIoU: best checkpoint only")

    if args.reuse_pretrain and os.path.isfile(paths["pretrain_ckpt"]):
        logger.info(f"Reuse existing pretrain checkpoint: {paths['pretrain_ckpt']}")
        return paths["pretrain_ckpt"]

    train_loader, val_loader = build_synth_loaders(args)

    ckpt, _ = train_one_stage(
        stage_name=f"{exp_name}_pretrain",
        train_loader=train_loader,
        val_loader=val_loader,
        logger=logger,
        save_path=paths["pretrain_ckpt"],
        epoch_num=args.synth_epochs,
        lr=args.synth_lr,
        weight_decay=args.synth_wd,
        args=args,
        use_dfe=exp_cfg["use_dfe"],
        use_sgmpp=exp_cfg["use_sgmpp"],
        load_from=None,
        skip_existing=False,
    )

    return ckpt


def run_finetune(
    args,
    exp_name: str,
    exp_cfg: Dict,
    paths: Dict[str, str],
    pretrain_ckpt: Optional[str],
) -> List[Dict]:
    logger = setup_logger(
        name=f"{exp_name}_finetune",
        log_file=paths["finetune_log_file"],
    )

    logger.info("=" * 100)
    logger.info(f"Stage 2: S2DS fine-tuning | {exp_name}")
    logger.info("=" * 100)
    logger.info(f"use_synth_pretrain = {exp_cfg['use_synth_pretrain']}")
    logger.info(f"use_dfe = {exp_cfg['use_dfe']}")
    logger.info(f"use_sgmpp = {exp_cfg['use_sgmpp']}")
    logger.info(f"pretrain_ckpt = {pretrain_ckpt}")
    logger.info(f"s2ds_root = {args.s2ds_root}")
    logger.info(f"folds = {args.folds}")
    logger.info(f"epochs = {args.finetune_epochs}")
    logger.info(f"lr = {args.finetune_lr}")
    logger.info(f"weight_decay = {args.finetune_wd}")
    logger.info("Epoch validation: P/R/F1/IoU only")
    logger.info("clIoU: best checkpoint only")

    rows = []

    for fold in args.folds:
        logger.info("\n" + "-" * 100)
        logger.info(f"Fine-tuning | {exp_name} | fold {fold}")
        logger.info("-" * 100)

        train_loader, val_loader = build_s2ds_loaders(args, fold)

        save_path = os.path.join(
            paths["finetune_weights_dir"],
            f"best_finetune_fold{fold}.pth",
        )

        _, best_metrics = train_one_stage(
            stage_name=f"{exp_name}_fold{fold}",
            train_loader=train_loader,
            val_loader=val_loader,
            logger=logger,
            save_path=save_path,
            epoch_num=args.finetune_epochs,
            lr=args.finetune_lr,
            weight_decay=args.finetune_wd,
            args=args,
            use_dfe=exp_cfg["use_dfe"],
            use_sgmpp=exp_cfg["use_sgmpp"],
            load_from=pretrain_ckpt,
            skip_existing=args.skip_existing,
        )

        row = {
            "experiment": exp_name,
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
            "use_synth_pretrain": exp_cfg["use_synth_pretrain"],
            "use_dfe": exp_cfg["use_dfe"],
            "use_sgmpp": exp_cfg["use_sgmpp"],
            "checkpoint": save_path,
        }

        rows.append(row)

    return rows


# ============================================================
# Args
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--synth-root", type=str, default=DEFAULT_SYNTH_ROOT)
    parser.add_argument("--s2ds-root", type=str, default=DEFAULT_S2DS_ROOT)
    parser.add_argument("--base-dir", type=str, default=DEFAULT_BASE_DIR)

    parser.add_argument("--backbone-path", type=str, default=None)

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

    parser.add_argument("--train-size", type=int, default=TRAIN_SIZE)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--num-workers", type=int, default=NUM_WORKERS)

    parser.add_argument("--synth-epochs", type=int, default=SYNTH_EPOCHS)
    parser.add_argument("--finetune-epochs", type=int, default=FINETUNE_EPOCHS)

    parser.add_argument("--synth-lr", type=float, default=SYNTH_LR)
    parser.add_argument("--finetune-lr", type=float, default=FINETUNE_LR)

    parser.add_argument("--synth-wd", type=float, default=SYNTH_WD)
    parser.add_argument("--finetune-wd", type=float, default=FINETUNE_WD)

    parser.add_argument("--eta-min", type=float, default=ETA_MIN)
    parser.add_argument("--grad-clip", type=float, default=GRAD_CLIP)

    parser.add_argument("--channel", type=int, default=CHANNEL)
    parser.add_argument("--num-queries", type=int, default=NUM_QUERIES)

    parser.add_argument("--threshold", type=float, default=THRESHOLD)

    parser.add_argument("--pos-crop-prob", type=float, default=POS_CROP_PROB)
    parser.add_argument("--dilation-kernel", type=int, default=DILATION_KERNEL)

    parser.add_argument(
        "--reuse-pretrain",
        action="store_true",
        help="Reuse existing synthetic pretrain checkpoint if available.",
    )

    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip existing fine-tune checkpoints and only evaluate them.",
    )

    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="Use deterministic CuDNN. Slower but more reproducible.",
    )

    return parser.parse_args()


# ============================================================
# Main
# ============================================================

def main():
    args = parse_args()

    set_seed(SEED)

    if args.deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    else:
        torch.backends.cudnn.benchmark = True

    ensure_dir(args.base_dir)
    ensure_dir(os.path.join(args.base_dir, "summary"))

    all_rows = []

    print("=" * 100)
    print("CrackMorphFormer training")
    print("=" * 100)
    print(f"base_dir        = {args.base_dir}")
    print(f"synth_root      = {args.synth_root}")
    print(f"s2ds_root       = {args.s2ds_root}")
    print(f"experiments     = {args.experiments}")
    print(f"folds           = {args.folds}")
    print(f"train_size      = {args.train_size}")
    print(f"batch_size      = {args.batch_size}")
    print(f"synth_epochs    = {args.synth_epochs}")
    print(f"finetune_epochs = {args.finetune_epochs}")
    print("epoch eval      = P/R/F1/IoU only")
    print("clIoU eval      = best checkpoint only")
    print("=" * 100)

    for exp_name in args.experiments:
        exp_cfg = EXPERIMENTS[exp_name]

        print("\n" + "=" * 100)
        print(f"Run experiment: {exp_name}")
        print("=" * 100)

        paths = build_exp_paths(args.base_dir, exp_name)

        pretrain_ckpt = None

        if exp_cfg["use_synth_pretrain"]:
            pretrain_ckpt = run_pretrain(
                args=args,
                exp_name=exp_name,
                exp_cfg=exp_cfg,
                paths=paths,
            )

        rows = run_finetune(
            args=args,
            exp_name=exp_name,
            exp_cfg=exp_cfg,
            paths=paths,
            pretrain_ckpt=pretrain_ckpt,
        )

        write_rows_csv(
            rows=rows,
            csv_path=paths["summary_csv"],
        )

        all_rows.extend(rows)

        global_csv = os.path.join(
            args.base_dir,
            "summary",
            "all_experiments_summary.csv",
        )

        write_rows_csv(
            rows=all_rows,
            csv_path=global_csv,
        )

        print_summary(all_rows)

        print(f"\nSaved experiment summary: {paths['summary_csv']}")
        print(f"Saved global summary: {global_csv}")

    print("\n" + "=" * 100)
    print("Training finished.")
    print("=" * 100)


if __name__ == "__main__":
    main()