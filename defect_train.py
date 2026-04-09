# -*- coding: utf-8 -*-
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os
import logging
from tqdm import tqdm
from torch.optim.lr_scheduler import CosineAnnealingLR
from model.CrackMorphFormer import CrackMorphFormer as MyModel
from ESDI_dataloader import get_loader

# ================= 全量实验从零启动配置 =================
EXP_NAME = "CrackMorphFormer" 
BASE_DIR = os.path.join("results", EXP_NAME)
WEIGHTS_DIR = os.path.join(BASE_DIR, "weights")
LOG_DIR = os.path.join(BASE_DIR, "txt")
LOG_FILE = os.path.join(LOG_DIR, "training_full_process.txt")

os.makedirs(WEIGHTS_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

def setup_logger(log_file):
    logger = logging.getLogger(EXP_NAME)
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        fh = logging.FileHandler(log_file, mode='w', encoding='utf-8')
        ch = logging.StreamHandler()
        formatter = logging.Formatter('%(message)s')
        fh.setFormatter(formatter); ch.setFormatter(formatter)
        logger.addHandler(fh); logger.addHandler(ch)
    return logger

logger = setup_logger(LOG_FILE)

# ----------------- 损失函数 -----------------
class BoundaryRelaxedLoss(nn.Module):
    def __init__(self, alpha=0.2, dilation_kernel=5, ignore_index=255):
        super(BoundaryRelaxedLoss, self).__init__()
        self.alpha = alpha; self.kernel_size = dilation_kernel; self.ignore_index = ignore_index  
    def forward(self, pred_logits, target):
        valid_mask = (target != self.ignore_index).float()
        clean_target = target.clone().float(); clean_target[target == self.ignore_index] = 0.0 
        padding = self.kernel_size // 2
        dilated_target = F.max_pool2d(clean_target, kernel_size=self.kernel_size, stride=1, padding=padding)
        m_relax = dilated_target - clean_target 
        bce_loss = F.binary_cross_entropy_with_logits(pred_logits, clean_target, reduction='none')
        weight_mask = torch.ones_like(bce_loss)
        weight_mask[(clean_target == 0) & (m_relax == 1)] = self.alpha
        return (bce_loss * weight_mask * valid_mask).sum() / (valid_mask.sum() + 1e-8)

def dice_loss(p_logits, target):
    pred = torch.sigmoid(p_logits); v_mask = (target != 255).float()
    c_gt = target.clone(); c_gt[target == 255] = 0.0
    intersection = (pred * v_mask * c_gt).sum()
    return 1 - (2.*intersection + 1e-6) / ((pred*v_mask).sum() + (c_gt*v_mask).sum() + 1e-6)

def compute_total_loss(preds, gts, criterion):
    total = 0; weights = [0.5, 0.7, 1.0] 
    for i, pred in enumerate(preds):
        total += weights[i] * (criterion(pred, gts) + dice_loss(pred, gts))
    return total

def eval_metrics(val_loader, model):
    model.eval(); total_tp, total_fp, total_fn, total_union = 0, 0, 0, 0
    with torch.no_grad():
        for data in val_loader:
            images, gts = data['image'].cuda(), data['label'].cuda() 
            res = model(images)[-1]
            preds = (torch.sigmoid(res) > 0.5).float()
            mask = (gts != 255)
            if not mask.any(): continue
            p_v, g_v = preds[mask], gts[mask]
            total_tp += (p_v * g_v).sum().item(); total_fp += (p_v * (1-g_v)).sum().item()
            total_fn += ((1-p_v) * g_v).sum().item(); total_union += (p_v + g_v).gt(0).sum().item()
    p = total_tp / (total_tp + total_fp + 1e-8); r = total_tp / (total_tp + total_fn + 1e-8)
    f1 = 2 * (p * r) / (p + r + 1e-8); iou = total_tp / (total_union + 1e-8)
    return p, r, f1, iou

def run_train_stage(stage, fold, synth_root, s2ds_root, pretrain_weights=None):
    train_size = 512; batch_size = 4; lr = 8e-5; epoch_num = 60
    net = MyModel(channel=64).cuda()
    best_weight_path = os.path.join(WEIGHTS_DIR, f"best_{stage}_fold{fold}.pth" if stage == "finetune" else "best_synth_pretrained.pth")

    if stage == "pretrain":
        logger.info(f"\n🚀 [Stage 1] 合成域从零重新训练开始...")
        loader = get_loader(synth_root, 'synthcrack', 'train', 1, batch_size, train_size)
        v_loader = get_loader(synth_root, 'synthcrack', 'val', 1, 1, train_size, shuffle=False)
        base_criterion = nn.BCEWithLogitsLoss(reduction='none')
        def loss_func(p, t):
            vm = (t != 255).float(); ct = t.clone(); ct[t == 255] = 0.0
            return (base_criterion(p, ct) * vm).sum() / (vm.sum() + 1e-8)
    else:
        logger.info(f"\n🔥 [Stage 2] S2DS 第 {fold} 折微调启动 (100% Patch 策略)...")
        loader = get_loader(s2ds_root, 's2ds5', 'train', fold, batch_size, train_size)
        v_loader = get_loader(s2ds_root, 's2ds5', 'val', fold, 1, train_size, shuffle=False)
        loss_func = BoundaryRelaxedLoss(alpha=0.2)
        if pretrain_weights: net.load_state_dict(torch.load(pretrain_weights)); logger.info(f"✅ 载入刚跑出的最佳预训练权重")

    optimizer = optim.Adam(net.parameters(), lr=lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=epoch_num, eta_min=1e-7)
    best_f1 = 0.0

    for ep in range(epoch_num):
        net.train(); l_epoch = 0.0
        pbar = tqdm(loader, desc=f"Fold {fold} Ep {ep+1}" if stage=="finetune" else f"Pretrain Ep {ep+1}", unit="batch", leave=False)
        for data in pbar:
            imgs, gts = data['image'].cuda(), data['label'].cuda()
            optimizer.zero_grad()
            loss = compute_total_loss(net(imgs), gts, loss_func)
            loss.backward(); optimizer.step(); l_epoch += loss.item()
        
        scheduler.step(); p, r, f1, iou = eval_metrics(v_loader, net)
        log_msg = f"📊 Ep {ep+1}: Loss:{l_epoch/len(loader):.4f} | P:{p:.4f} R:{r:.4f} F1:{f1:.4f} IoU:{iou:.4f}"
        if f1 > best_f1:
            torch.save(net.state_dict(), best_weight_path); best_f1 = f1
            log_msg += f"  🌟 New Best!"
        logger.info(log_msg)
    return best_weight_path

def main():
    SYNTH_ROOT = "/home/skye/data/Skye/databases/synthcrack"
    S2DS_ROOT = "/home/skye/data/Skye/databases/s2ds5"
    logger.info(">>> 启动全量从零训练：CrackMorphFormer架构 <<<")
    # 步骤 1：合成域预训练
    new_best_pretrain = run_train_stage("pretrain", 1, SYNTH_ROOT, S2DS_ROOT)
    # 步骤 2：真实域五折微调
    for fold in range(1, 6):
        run_train_stage("finetune", fold, SYNTH_ROOT, S2DS_ROOT, pretrain_weights=new_best_pretrain)

if __name__ == '__main__': main()