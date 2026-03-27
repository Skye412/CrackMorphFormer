import numpy as np
import torch
from torch.autograd import Variable
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch.optim as optim
import os
import logging
import sys
import torch.nn.functional as F
from PIL import Image
import cv2
import time

from sod_metrics import MAE, Emeasure, Fmeasure, Smeasure, WeightedFmeasure
from torch.optim.lr_scheduler import CosineAnnealingLR
from model.WPFormer import WPFormer
from ESDI_dataloader import get_loader

# ================= 新增：边界松弛损失函数 =================
class BoundaryRelaxedLoss(nn.Module):
    def __init__(self, alpha=0.1, dilation_kernel=5, ignore_index=255):
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
        return final_loss.sum() / (valid_mask.sum() + 1e-6)

# =========================================================

def eval_psnr(test_image_root, test_gt_root, train_size, model):
    FM = Fmeasure()
    WFM = WeightedFmeasure()
    SM = Smeasure()

    img_transform = transforms.Compose([
        transforms.Resize((384, 384)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    images = [os.path.join(test_image_root, f) for f in os.listdir(test_image_root)]
    gts = [os.path.join(test_gt_root, p) for p in os.listdir(test_gt_root)]
    images = sorted(images)
    gts = sorted(gts)
    
    model.eval()
    for index in range(len(images)):
        ori_image = Image.open(images[index]).convert("RGB")
        image = img_transform(ori_image).unsqueeze(0).cuda()
        gt = cv2.imread(gts[index], cv2.IMREAD_GRAYSCALE)
        H, W = gt.shape
        
        with torch.no_grad():
            predictions_mask = model(image)
            res = predictions_mask[-1]

        res = torch.sigmoid(res).data.cpu().numpy().squeeze()
        pred = (res - res.min()) / (res.max() - res.min() + 1e-8)
        pred = Image.fromarray(pred * 255).convert("L")
        pred = pred.resize((W, H), resample=Image.BILINEAR)
        pred = np.array(pred)

        FM.step(pred=pred, gt=gt)
        WFM.step(pred=pred, gt=gt)
        SM.step(pred=pred, gt=gt)
        
    fm = FM.get_results()["fm"]
    wfm = WFM.get_results()["wfm"]
    sm = SM.get_results()["sm"]

    return fm["curve"].max(), wfm

def total_loss(pred, mask, criterion):
    """ 更新的 Loss 计算机制 """
    # 预测结果未经过 sigmoid，直接输入到 criterion 中 (因为使用了 with_logits)
    bce = criterion(pred, mask)
    
    # IoU Loss (这里为了兼容 ignore_index 做了简单掩码处理)
    pred_sig = torch.sigmoid(pred)
    valid_mask = (mask != 255).float()
    clean_mask = mask.clone()
    clean_mask[mask == 255] = 0.0
    
    inter = (pred_sig * clean_mask * valid_mask).sum(dim=(2, 3))
    union = ((pred_sig + clean_mask) * valid_mask).sum(dim=(2, 3))
    iou = 1 - inter / (union - inter + 1e-6)
    
    return bce + iou.mean()

def train(dataset_name):
    # ================= 科研开关配置 =================
    STAGE = "finetune"  # 选项: "pretrain" (合成数据) 或 "finetune" (真实 S2DS)
    # ===============================================

    epoch_num = 100
    epoch_val = 10
    train_size = 384
    batch_size = 8

    net = WPFormer(method="pvt_v2_b2", channel=64)
    if torch.cuda.is_available():
        net = net.cuda()

    file_dir = ".\datasets\\"
    train_image_root = os.path.join(file_dir, dataset_name, "train", "images")
    train_gt_root = os.path.join(file_dir, dataset_name, "train", "gt")
    test_image_root = os.path.join(file_dir, dataset_name, "test", "images")
    test_gt_root = os.path.join(file_dir, dataset_name, "test", "gt")

    train_loader1 = get_loader(train_image_root, train_gt_root, batchsize=batch_size, trainsize=train_size, is_train=True)

    optimizer = optim.Adam(net.parameters(), lr=8e-5)
    lr_scheduler = CosineAnnealingLR(optimizer, T_max=epoch_num, eta_min=1e-7)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 根据双阶段自动配置 Loss
    if STAGE == "pretrain":
        print("🚀 [Stage 1] 启动合成数据预训练 (BCE Loss)")
        criterion = nn.BCEWithLogitsLoss()
    else:
        print("🔥 [Stage 2] 启动真实数据抗噪微调 (Boundary-Relaxed Loss)")
        criterion = BoundaryRelaxedLoss(alpha=0.1, dilation_kernel=5, ignore_index=255)
        # TODO: 这里填入你跑出来的合成数据最佳权重的路径！
        # net.load_state_dict(torch.load('save/best_synth_pretrained.pth'))

    print("---start training...")
    best_fm = 0

    for epoch in range(epoch_num):
        start_time = time.time()
        running_loss = 0.0

        for i, data in enumerate(train_loader1):
            inputs, labels = data['image'], data['label']
            images, gts = inputs.to(device), labels.to(device)
            
            # gts 处理：如果是 255 则保留用于 Loss 中的忽略
            # 注意 DataLoader 吐出的是 [0,1] 的 tensor (如果不处理 255 的话)，
            # 如果你的原图 mask 裂缝是 255，那么这里由于 transforms.ToTensor 变成了 1.0。
            # 如果你想使用 255 隔离，需在数据加载阶段小心处理。
            
            optimizer.zero_grad()
            predictions_mask = net(images)

            mask_losses = 0
            for pred in predictions_mask:
                mask_losses = mask_losses + total_loss(pred, gts, criterion)

            mask_losses.backward()
            optimizer.step()
            running_loss += mask_losses.item()

        end_time = time.time()
        print(f'Epoch [{epoch}/{epoch_num}] Loss: {running_loss/len(train_loader1):.4f} Cost: {end_time - start_time:.2f}s')

        lr_scheduler.step()

        # 每 epoch_val 评估一次
        if (epoch + 1) % epoch_val == 0 or (epoch + 1) == epoch_num:
            fm, wfm = eval_psnr(test_image_root, test_gt_root, train_size, net)

            if fm > best_fm:
                save_path = os.path.join(".\save", dataset_name)
                os.makedirs(save_path, exist_ok=True)
                torch.save(net.state_dict(), os.path.join(save_path, f"{STAGE}_best_F1_{fm:.4f}.pth"))
                best_fm = fm
            
            print(f"--> Eval: maxFm: {fm:.4f}, wFm: {wfm:.4f} | Best maxFm: {best_fm:.4f}")
            net.train()

if __name__ == '__main__':
    # 你的 S2DS 数据集名称
    dataset_name = "S2DS" 
    train(dataset_name)