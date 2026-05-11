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

# 使用支持消融的模型
from model.CrackMorphFormer_ablation import CrackMorphFormer as MyModel
from ESDI_dataloader import get_loader


# ============================================================
# Default paths
# ============================================================
DEFAULT_SYNTH_ROOT = "/home/skye/data/Skye/databases/synthcrack"
DEFAULT_S2DS_ROOT = "/home/skye/data/Skye/databases/s2ds5"
DEFAULT_BASE_DIR = "results/ablation"

# Training settings
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
    os.makedirs(os.path.join(base_dir, "finetune", "weights"), exist_ok=True)
    os.makedirs(os.path.join(base_dir, "finetune", "txt"), exist_ok=True)
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
        "precision": value, "recall": value, "f1": value, "iou": value,
        "clIoU@2": value, "clIoU@4": value, "clIoU@6": value, "clIoU@8": value,
        "best_epoch": int(value),
    }

def compute_loss(pred_logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    valid_mask = (target != 255).float()
    clean_target = target.clone()
    clean_target[target == 255] = 0.0
    bce = F.binary_cross_entropy_with_logits(pred_logits, clean_target, reduction="none")
    bce = (bce * valid_mask).sum() / (valid_mask.sum() + 1e-8)
    pred = torch.sigmoid(pred_logits)
    inter = (pred * valid_mask * clean_target).sum()
    dice = 1.0 - (2.0 * inter + 1e-6) / ((pred * valid_mask).sum() + (clean_target * valid_mask).sum() + 1e-6)
    return bce + dice

def update_cl_stats(cl_stats, pred_bin, gt_bin, taus):
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
def eval_metrics(val_loader, model, threshold=THRESHOLD, taus=CLI0U_TAUS):
    model.eval()
    total_tp = total_fp = total_fn = total_union = 0.0
    cl_stats = {tau: {"tp": 0.0, "fp": 0.0, "fn": 0.0} for tau in taus}
    for data in val_loader:
        images = data["image"].cuda(non_blocking=True)
        gts = data["label"].cuda(non_blocking=True).float()
        output = model(images)
        logits = output[-1] if isinstance(output, (list, tuple)) else output
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
            update_cl_stats(cl_stats, pred_bin, gt_bin, taus)
    precision = total_tp / (total_tp + total_fp + 1e-8)
    recall = total_tp / (total_tp + total_fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    iou = total_tp / (total_union + 1e-8)
    metrics = {"precision": precision, "recall": recall, "f1": f1, "iou": iou}
    for tau in taus:
        tp = cl_stats[tau]["tp"]
        fp = cl_stats[tau]["fp"]
        fn = cl_stats[tau]["fn"]
        metrics[f"clIoU@{tau}"] = tp / (tp + fp + fn + 1e-8)
    return metrics

# ======================== Data loaders ========================
def build_synth_loaders(synth_root, batch_size, train_size, num_workers):
    train_loader = get_loader(synth_root, "synthcrack", "train", 1,
                              batchsize=batch_size, trainsize=train_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True,
                              pos_crop_prob=POS_CROP_PROB, dilation_kernel=DILATION_KERNEL)
    val_loader = get_loader(synth_root, "synthcrack", "val", 1,
                            batchsize=1, trainsize=train_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True,
                            pos_crop_prob=POS_CROP_PROB, dilation_kernel=DILATION_KERNEL)
    return train_loader, val_loader

def build_s2ds_loaders(s2ds_root, fold, batch_size, train_size, num_workers):
    train_loader = get_loader(s2ds_root, "s2ds5", "train", fold,
                              batchsize=batch_size, trainsize=train_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True,
                              pos_crop_prob=POS_CROP_PROB, dilation_kernel=DILATION_KERNEL)
    val_loader = get_loader(s2ds_root, "s2ds5", "val", fold,
                            batchsize=1, trainsize=train_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True,
                            pos_crop_prob=POS_CROP_PROB, dilation_kernel=DILATION_KERNEL)
    return train_loader, val_loader

# ======================== Training helpers ========================
def train_one_stage(stage_name, train_loader, val_loader, logger, save_path,
                    epoch_num, lr, weight_decay, load_from=None, skip_existing=False,
                    use_dfe=True, use_astp=True):
    if skip_existing and os.path.isfile(save_path):
        model = MyModel(channel=64, use_dfe=use_dfe, use_astp=use_astp).cuda()
        model.load_state_dict(torch.load(save_path, map_location="cpu"), strict=True)
        metrics = eval_metrics(val_loader, model)
        metrics["best_epoch"] = -1
        logger.info(f"Skip training, evaluate existing: P={metrics['precision']:.4f} R={metrics['recall']:.4f} F1={metrics['f1']:.4f}")
        del model
        return save_path, metrics

    model = MyModel(channel=64, use_dfe=use_dfe, use_astp=use_astp).cuda()
    if load_from:
        state = torch.load(load_from, map_location="cpu")
        model.load_state_dict(state, strict=True)
        logger.info(f"Loaded weights from {load_from}")

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=epoch_num, eta_min=ETA_MIN)

    best_f1 = -1.0
    best_metrics = default_metric_dict(0.0)
    for ep in range(epoch_num):
        model.train()
        epoch_loss = 0.0
        pbar = tqdm(train_loader, desc=f"{stage_name} Ep {ep+1}/{epoch_num}", leave=False)
        for data in pbar:
            imgs = data["image"].cuda(non_blocking=True)
            gts = data["label"].cuda(non_blocking=True).float()
            optimizer.zero_grad(set_to_none=True)
            outputs = model(imgs)
            if not isinstance(outputs, (list, tuple)):
                outputs = [outputs]
            loss = sum(DS_WEIGHTS[i] * compute_loss(logit, gts) for i, logit in enumerate(outputs) if i < len(DS_WEIGHTS))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_loss += loss.item()
        scheduler.step()
        metrics = eval_metrics(val_loader, model)
        f1 = metrics["f1"]
        logger.info(f"[{stage_name}] Ep {ep+1}: Loss={epoch_loss/len(train_loader):.4f} P={metrics['precision']:.4f} R={metrics['recall']:.4f} F1={f1:.4f} IoU={metrics['iou']:.4f} clIoU@4={metrics['clIoU@4']:.4f}")
        if f1 > best_f1:
            best_f1 = f1
            best_metrics = metrics.copy()
            best_metrics["best_epoch"] = ep + 1
            torch.save(model.state_dict(), save_path)
            logger.info(" -> Best model saved")
    logger.info(f"Best {stage_name}: epoch {best_metrics['best_epoch']} F1={best_metrics['f1']:.4f}")
    del model
    torch.cuda.empty_cache()
    return save_path, best_metrics

# ======================== Main ablation flow ========================
def run_ablation(args):
    # 根据 ablation 模式确定模型开关和是否使用合成预训练
    if args.ablation == "full":
        use_dfe, use_astp, use_synth = True, True, True
        variant_name = "Full"
    elif args.ablation == "no_synth":
        use_dfe, use_astp, use_synth = True, True, False
        variant_name = "RealOnly"
    elif args.ablation == "no_dfe":
        use_dfe, use_astp, use_synth = False, True, True
        variant_name = "No_DFE"
    elif args.ablation == "no_astp":
        use_dfe, use_astp, use_synth = True, False, True
        variant_name = "No_ASTP"
    elif args.ablation == "no_both":
        use_dfe, use_astp, use_synth = False, False, True
        variant_name = "No_DFE_ASTP"
    else:
        raise ValueError("Invalid ablation mode")

    base_dir = os.path.join(args.base_dir, variant_name)
    ensure_dirs(base_dir)
    paths = {
        "shared_pretrain_ckpt": os.path.join(base_dir, "shared_pretrain", "weights", "best_pretrain.pth"),
        "shared_pretrain_log": os.path.join(base_dir, "shared_pretrain", "txt", "log.txt"),
        "finetune_weights_dir": os.path.join(base_dir, "finetune", "weights"),
        "finetune_log": os.path.join(base_dir, "finetune", "txt", "log.txt"),
        "summary_csv": os.path.join(base_dir, "summary", "results.csv"),
    }
    os.makedirs(paths["finetune_weights_dir"], exist_ok=True)

    # Stage 1: Synthetic pretraining (only if use_synth is True)
    pretrain_weights = None
    if use_synth:
        logger = setup_logger("pretrain", paths["shared_pretrain_log"])
        logger.info("=== Synthetic pretraining ===")
        train_loader, val_loader = build_synth_loaders(
            args.synth_root, args.batch_size, args.train_size, args.num_workers)
        ckpt, _ = train_one_stage(
            stage_name="pretrain",
            train_loader=train_loader,
            val_loader=val_loader,
            logger=logger,
            save_path=paths["shared_pretrain_ckpt"],
            epoch_num=args.synth_epochs,
            lr=args.synth_lr,
            weight_decay=args.synth_wd,
            load_from=None,
            skip_existing=args.skip_existing,
            use_dfe=use_dfe,
            use_astp=use_astp,
        )
        pretrain_weights = ckpt
    else:
        print("=== Skipping synthetic pretraining (real-only) ===")

    # Stage 2: Fine-tuning on S2DS
    logger = setup_logger("finetune", paths["finetune_log"])
    logger.info(f"=== Fine-tuning on S2DS (variant: {variant_name}) ===")
    rows = []
    for fold in args.folds:
        logger.info(f"\n--- Fold {fold} ---")
        train_loader, val_loader = build_s2ds_loaders(
            args.s2ds_root, fold, args.batch_size, args.train_size, args.num_workers)
        save_path = os.path.join(paths["finetune_weights_dir"], f"best_fold{fold}.pth")
        _, best_metrics = train_one_stage(
            stage_name=f"finetune_fold{fold}",
            train_loader=train_loader,
            val_loader=val_loader,
            logger=logger,
            save_path=save_path,
            epoch_num=args.finetune_epochs,
            lr=args.finetune_lr,
            weight_decay=args.finetune_wd,
            load_from=pretrain_weights,
            skip_existing=args.skip_existing,
            use_dfe=use_dfe,
            use_astp=use_astp,
        )
        row = {"variant": variant_name, "fold": fold, "best_epoch": best_metrics["best_epoch"]}
        for k in ["precision", "recall", "f1", "iou", "clIoU@2", "clIoU@4", "clIoU@6", "clIoU@8"]:
            row[k] = best_metrics[k]
        row["checkpoint"] = save_path
        rows.append(row)

    # Write summary
    with open(paths["summary_csv"], "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["variant","fold","best_epoch","precision","recall","f1","iou",
                                               "clIoU@2","clIoU@4","clIoU@6","clIoU@8","checkpoint"])
        writer.writeheader()
        for r in rows:
            writer.writerow(r)
    print(f"\nResults saved to {paths['summary_csv']}")
    return rows

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--synth-root", type=str, default=DEFAULT_SYNTH_ROOT)
    parser.add_argument("--s2ds-root", type=str, default=DEFAULT_S2DS_ROOT)
    parser.add_argument("--base-dir", type=str, default=DEFAULT_BASE_DIR)
    parser.add_argument("--folds", nargs="+", type=int, default=[1,2,3,4,5])
    parser.add_argument("--train-size", type=int, default=TRAIN_SIZE)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--synth-epochs", type=int, default=SYNTH_EPOCHS)
    parser.add_argument("--finetune-epochs", type=int, default=FINETUNE_EPOCHS)
    parser.add_argument("--synth-lr", type=float, default=SYNTH_LR)
    parser.add_argument("--finetune-lr", type=float, default=FINETUNE_LR)
    parser.add_argument("--synth-wd", type=float, default=SYNTH_WD)
    parser.add_argument("--finetune-wd", type=float, default=FINETUNE_WD)
    parser.add_argument("--reuse-pretrain", action="store_true")          # 仅用于 full 模式
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--ablation", type=str, required=True,
                        choices=["full", "no_synth", "no_dfe", "no_astp", "no_both"],
                        help="Ablation mode")
    return parser.parse_args()

def main():
    args = parse_args()
    set_seed(SEED)
    torch.backends.cudnn.benchmark = True
    print("="*100)
    print(f"Starting ablation: {args.ablation}")
    print("="*100)
    _ = run_ablation(args)

if __name__ == "__main__":
    main()