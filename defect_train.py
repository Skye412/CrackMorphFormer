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

# 实验配置
EXP_NAME = "CrackMorphFormer_SmallSample_Refined"
BASE_DIR = os.path.join("results", EXP_NAME)
WEIGHTS_DIR = os.path.join(BASE_DIR, "weights")
LOG_DIR = os.path.join(BASE_DIR, "txt")
LOG_FILE = os.path.join(LOG_DIR, "training_log.txt")

os.makedirs(WEIGHTS_DIR, exist_ok=True); os.makedirs(LOG_DIR, exist_ok=True)

def setup_logger(log_file):
    logger = logging.getLogger(EXP_NAME)
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        fh = logging.FileHandler(log_file, mode='w', encoding='utf-8')
        ch = logging.StreamHandler()
        fh.setFormatter(logging.Formatter('%(message)s'))
        logger.addHandler(fh); logger.addHandler(ch)
    return logger

logger = setup_logger(LOG_FILE)

# 损失函数 (Masked)
def compute_loss(pred_logits, target):
    valid_mask = (target != 255).float()
    clean_target = target.clone(); clean_target[target == 255] = 0.0
    
    # BCE
    bce = F.binary_cross_entropy_with_logits(pred_logits, clean_target, reduction='none')
    bce = (bce * valid_mask).sum() / (valid_mask.sum() + 1e-8)
    
    # Dice
    pred = torch.sigmoid(pred_logits)
    inter = (pred * valid_mask * clean_target).sum()
    dice = 1 - (2.*inter + 1e-6) / ((pred*valid_mask).sum() + (clean_target*valid_mask).sum() + 1e-6)
    
    return bce + dice

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
    train_size = 512; batch_size = 4
    # 🔧 策略微调：真实域降低学习率，增加正则化
    lr = 8e-5 if stage == "pretrain" else 5e-5
    wd = 1e-4 if stage == "pretrain" else 5e-4
    epoch_num = 30 if stage == "pretrain" else 60
    
    net = MyModel(channel=64).cuda()
    best_weight_path = os.path.join(WEIGHTS_DIR, f"best_{stage}_fold{fold}.pth")

    if stage == "pretrain":
        logger.info("\n🚀 [Stage 1] 合成域预训练 (5000张) 开始...")
        loader = get_loader(synth_root, 'synthcrack', 'train', 1, batch_size, train_size)
        v_loader = get_loader(synth_root, 'synthcrack', 'val', 1, 1, train_size, shuffle=False)
    else:
        logger.info(f"\n🔥 [Stage 2] S2DS 第 {fold} 折微调 (231张) 开始...")
        loader = get_loader(s2ds_root, 's2ds5', 'train', fold, batch_size, train_size)
        v_loader = get_loader(s2ds_root, 's2ds5', 'val', fold, 1, train_size, shuffle=False)
        if pretrain_weights: net.load_state_dict(torch.load(pretrain_weights))

    optimizer = optim.AdamW(net.parameters(), lr=lr, weight_decay=wd)
    scheduler = CosineAnnealingLR(optimizer, T_max=epoch_num, eta_min=1e-7)
    best_f1 = 0.0

    for ep in range(epoch_num):
        net.train(); l_epoch = 0.0
        for data in tqdm(loader, desc=f"{stage.capitalize()} Ep {ep+1}", leave=False):
            imgs, gts = data['image'].cuda(), data['label'].cuda()
            optimizer.zero_grad()
            preds = net(imgs); loss = 0
            for i, p_logit in enumerate(preds): loss += [0.5, 0.7, 1.0][i] * compute_loss(p_logit, gts)
            loss.backward(); optimizer.step(); l_epoch += loss.item()
        
        scheduler.step(); p, r, f1, iou = eval_metrics(v_loader, net)
        log_msg = f"📊 Ep {ep+1}: Loss:{l_epoch/len(loader):.4f} | P:{p:.4f} R:{r:.4f} F1:{f1:.4f} IoU:{iou:.4f}"
        if f1 > best_f1:
            torch.save(net.state_dict(), best_weight_path); best_f1 = f1; log_msg += " 🌟 Best!"
        logger.info(log_msg)
    return best_weight_path

def main():
    SYNTH_ROOT = "/home/skye/data/Skye/databases/synthcrack"
    S2DS_ROOT = "/home/skye/data/Skye/databases/s2ds5"
    logger.info(">>> 启动针对小样本优化的全量训练流程 <<<")
    
    new_best_pretrain = run_train_stage("pretrain", 1, SYNTH_ROOT, S2DS_ROOT)
    for fold in range(1, 6):
        run_train_stage("finetune", fold, SYNTH_ROOT, S2DS_ROOT, pretrain_weights=new_best_pretrain)

if __name__ == '__main__': main()