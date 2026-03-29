import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os
import time
from tqdm import tqdm

from torch.optim.lr_scheduler import CosineAnnealingLR
from model.WPFormer import WPFormer
from ESDI_dataloader import get_loader

# ================= 1. 边界松弛损失函数 =================
class BoundaryRelaxedLoss(nn.Module):
    def __init__(self, alpha=0.2, dilation_kernel=5, ignore_index=255): # Alpha 提至 0.2，减少过分豁免
        super(BoundaryRelaxedLoss, self).__init__()
        self.alpha = alpha                
        self.kernel_size = dilation_kernel
        self.ignore_index = ignore_index  

    def forward(self, pred_logits, target):
        valid_mask = (target != self.ignore_index).float()
        clean_target = target.clone().float()
        clean_target[target == self.ignore_index] = 0.0 
        
        padding = self.kernel_size // 2
        dilated_target = F.max_pool2d(clean_target, kernel_size=self.kernel_size, stride=1, padding=padding)
        m_relax = dilated_target - clean_target 
        
        bce_loss = F.binary_cross_entropy_with_logits(pred_logits, clean_target, reduction='none')
        
        weight_mask = torch.ones_like(bce_loss)
        relax_condition = (clean_target == 0) & (m_relax == 1)
        weight_mask[relax_condition] = self.alpha
        
        final_loss = bce_loss * weight_mask * valid_mask
        return final_loss.sum() / (valid_mask.sum() + 1e-8)

# ================= 2. Dice Loss (形态约束提分神器) =================
def dice_loss(pred_logits, target, smooth=1e-6):
    pred = torch.sigmoid(pred_logits)
    # 忽略 255
    valid_mask = (target != 255).float()
    clean_target = target.clone()
    clean_target[target == 255] = 0.0
    
    pred = pred * valid_mask
    clean_target = clean_target * valid_mask
    
    intersection = (pred * clean_target).sum()
    dice = (2. * intersection + smooth) / (pred.sum() + clean_target.sum() + smooth)
    return 1 - dice

# ================= 3. 权重平衡与混合损失 =================
def compute_total_loss(preds, gts, criterion):
    total_loss_val = 0
    # 多尺度深度监督：越大的特征图（高分辨率）权重越高
    weights = [0.5, 0.7, 1.0] 
    
    for i, pred in enumerate(preds):
        # 基础分类 Loss (Stage1是BCE, Stage2是边界松弛)
        bce_part = criterion(pred, gts)
        # 结构形态 Loss
        dice_part = dice_loss(pred, gts)
        
        total_loss_val += weights[i] * (bce_part + dice_part)
        
    return total_loss_val

# ================= 4. 全面评估指标 =================
def eval_metrics(val_loader, model):
    model.eval()
    total_tp, total_fp, total_fn, total_union = 0, 0, 0, 0

    with torch.no_grad():
        for data in val_loader:
            images, gts = data['image'].cuda(), data['label'].cuda() 

            outputs = model(images)
            res = outputs[-1]
            preds = (torch.sigmoid(res) > 0.5).float()
            
            mask = (gts != 255)
            if not mask.any(): continue
            
            p_v, g_v = preds[mask], gts[mask]

            tp = (p_v * g_v).sum().item()
            fp = (p_v * (1 - g_v)).sum().item()
            fn = ((1 - p_v) * g_v).sum().item()
            
            total_tp += tp
            total_fp += fp
            total_fn += fn
            total_union += (tp + fp + fn)

    precision = total_tp / (total_tp + total_fp + 1e-8)
    recall = total_tp / (total_tp + total_fn + 1e-8)
    f1_score = 2 * (precision * recall) / (precision + recall + 1e-8)
    iou = total_tp / (total_union + 1e-8)

    return precision, recall, f1_score, iou

# ================= 5. 单阶段训练逻辑 =================
def run_train_stage(stage, fold, synth_root, s2ds_root, pretrain_weights=None):
    train_size = 512
    batch_size = 4
    lr = 8e-5
    # 按照你的建议：合成预训练延长至 60 轮！
    epoch_num = 60 if stage == "pretrain" else 60

    net = WPFormer(channel=64).cuda()

    if stage == "pretrain":
        print(f"\n{'='*20} [Stage 1: 合成数据预训练] {'='*20}")
        loader = get_loader(synth_root, 'synthcrack', 'train', 1, batch_size, train_size)
        v_loader = get_loader(synth_root, 'synthcrack', 'val', 1, 1, train_size, shuffle=False)
        # Stage 1 严厉监督
        criterion = nn.BCEWithLogitsLoss(reduction='none')
        # 封装一个兼容函数处理 255 忽略逻辑
        def bce_with_ignore(pred, target):
            valid_mask = (target != 255).float()
            clean_target = target.clone(); clean_target[target == 255] = 0.0
            loss = criterion(pred, clean_target) * valid_mask
            return loss.sum() / (valid_mask.sum() + 1e-8)
        loss_func = bce_with_ignore
    else:
        print(f"\n{'='*20} [Stage 2: S2DS 第 {fold} 折抗噪微调] {'='*20}")
        loader = get_loader(s2ds_root, 's2ds5', 'train', fold, batch_size, train_size)
        v_loader = get_loader(s2ds_root, 's2ds5', 'val', fold, 1, train_size, shuffle=False)
        loss_func = BoundaryRelaxedLoss(alpha=0.2, dilation_kernel=5, ignore_index=255)
        
        if pretrain_weights and os.path.exists(pretrain_weights):
            net.load_state_dict(torch.load(pretrain_weights))
            print(f"✅ 已加载预训练先验: {pretrain_weights}")

    optimizer = optim.Adam(net.parameters(), lr=lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=epoch_num, eta_min=1e-7)

    best_f1 = 0.0
    best_weight_path = f"save/best_{stage}_fold{fold}.pth" if stage == "finetune" else "save/best_synth_pretrained.pth"
    os.makedirs("save", exist_ok=True)

    for ep in range(epoch_num):
        net.train()
        l_epoch = 0.0
        pbar = tqdm(loader, desc=f"Epoch {ep+1}/{epoch_num}", unit="batch")
        
        for data in pbar:
            imgs, gts = data['image'].cuda(), data['label'].cuda()
            
            optimizer.zero_grad()
            preds = net(imgs)
            # 引入混合多尺度 Loss (BCE/BRL + Dice)
            loss = compute_total_loss(preds, gts, loss_func)
            
            loss.backward()
            optimizer.step()
            
            l_epoch += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        scheduler.step()

        p, r, f1, iou = eval_metrics(v_loader, net)
        print(f"📊 [Result] Ep {ep+1}: Loss:{l_epoch/len(loader):.4f} | P:{p:.4f} R:{r:.4f} F1:{f1:.4f} IoU:{iou:.4f}")

        if f1 > best_f1:
            torch.save(net.state_dict(), best_weight_path)
            best_f1 = f1
            print(f"🌟 New Best F1: {best_f1:.4f} (已存至 {best_weight_path})")

    return best_weight_path

def main():
    SYNTH_ROOT = "/home/skye/data/Skye/databases/synthcrack"
    S2DS_ROOT = "/home/skye/data/Skye/databases/s2ds5"
    
    best_pretrain_weight = run_train_stage("pretrain", 1, SYNTH_ROOT, S2DS_ROOT)
    for fold in range(1, 6):
        run_train_stage("finetune", fold, SYNTH_ROOT, S2DS_ROOT, pretrain_weights=best_pretrain_weight)

if __name__ == '__main__':
    main()