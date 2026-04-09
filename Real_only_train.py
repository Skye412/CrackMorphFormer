# -*- coding: utf-8 -*-
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

EXP_NAME = "CrackMorphFormer_RealOnly_Ablation"
BASE_DIR = os.path.join("results", EXP_NAME)
WEIGHTS_DIR = os.path.join(BASE_DIR, "weights")
LOG_DIR = os.path.join(BASE_DIR, "txt")
os.makedirs(WEIGHTS_DIR, exist_ok=True); os.makedirs(LOG_DIR, exist_ok=True)

def setup_logger():
    logger = logging.getLogger(EXP_NAME); logger.setLevel(logging.INFO)
    fh = logging.FileHandler(os.path.join(LOG_DIR, "train_log.txt"), mode='w', encoding='utf-8')
    ch = logging.StreamHandler(); formatter = logging.Formatter('%(message)s')
    fh.setFormatter(formatter); ch.setFormatter(formatter)
    logger.addHandler(fh); logger.addHandler(ch); return logger

logger = setup_logger()

class BenchmarkLoss(nn.Module):
    """对齐 WPFormer 的标准 Loss"""
    def __init__(self, ignore_index=255):
        super().__init__()
        self.ignore_index = ignore_index
    def dice_loss(self, p, gt):
        p = torch.sigmoid(p); mask = (gt != self.ignore_index).float()
        clean_gt = gt.clone().float(); clean_gt[gt == self.ignore_index] = 0.0
        inter = (p * mask * clean_gt).sum()
        return 1 - (2. * inter + 1e-6) / ((p * mask).sum() + (clean_gt * mask).sum() + 1e-6)
    def forward(self, preds, gt):
        mask = (gt != self.ignore_index).float()
        clean_gt = gt.clone().float(); clean_gt[gt == self.ignore_index] = 0.0
        total = 0; weights = [0.5, 0.7, 1.0]
        for i, p in enumerate(preds):
            bce = F.binary_cross_entropy_with_logits(p, clean_gt, reduction='none')
            total += weights[i] * ((bce * mask).sum() / (mask.sum() + 1e-8) + self.dice_loss(p, gt))
        return total

def eval_metrics(val_loader, model):
    model.eval(); tp, fp, fn, union = 0, 0, 0, 0
    with torch.no_grad():
        for data in val_loader:
            img, gt = data['image'].cuda(), data['label'].cuda()
            res = model(img)[-1]; pred = (torch.sigmoid(res) > 0.5).float()
            mask = (gt != 255); p_v, g_v = pred[mask], gt[mask]
            tp += (p_v * g_v).sum().item(); fp += (p_v * (1-g_v)).sum().item()
            fn += ((1-p_v) * g_v).sum().item(); union += (p_v + g_v).gt(0).sum().item()
    p = tp / (tp + fp + 1e-8); r = tp / (tp + fn + 1e-8)
    return p, r, 2*p*r/(p+r+1e-8), tp/(union+1e-8)

def run_train(fold, root):
    train_size = 512; batch_size = 8; lr = 8e-5; epochs = 60
    net = MyModel().cuda()
    loader = get_loader(root, 's2ds5', 'train', fold, batch_size, train_size)
    v_loader = get_loader(root, 's2ds5', 'val', fold, 1, train_size, shuffle=False)
    criterion = BenchmarkLoss()
    optimizer = optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
    
    logger.info(f"\n🚀 Fold {fold} 启动 (Real-Only | 对齐模式)")
    best_f1 = 0.0
    for ep in range(epochs):
        net.train(); train_l = 0.0
        for data in tqdm(loader, desc=f"Ep {ep+1}", leave=False):
            img, gt = data['image'].cuda(), data['label'].cuda()
            optimizer.zero_grad()
            loss = criterion(net(img), gt)
            loss.backward(); optimizer.step(); train_l += loss.item()
        
        scheduler.step(); p, r, f1, iou = eval_metrics(v_loader, net)
        # 记录 MPP 分支的门控权重
        gs = [d.gamma.item() for d in net.decoders]
        msg = f"📊 Ep {ep+1}: Loss:{train_l/len(loader):.4f} | F1:{f1:.4f} | IoU:{iou:.4f} | Gamma:{[f'{g:.3f}' for g in gs]}"
        if f1 > best_f1:
            best_f1 = f1; torch.save(net.state_dict(), os.path.join(WEIGHTS_DIR, f"best_f{fold}.pth"))
            msg += " 🌟"
        logger.info(msg)

if __name__ == '__main__':
    S2DS_PATH = "/home/skye/data/Skye/databases/s2ds5"
    for f in range(1, 6): run_train(f, S2DS_PATH)