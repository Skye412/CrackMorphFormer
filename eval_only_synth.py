import os
import torch
import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
from model.WPFormer import WPFormer

# ================= 配置区 =================
# 权重路径：指向你刚跑完的 Stage 1 最佳权重
MODEL_PATH = "save/best_synth_pretrained.pth"
# 真实测试数据集路径
S2DS_IMG_DIR = "/home/skye/data/Skye/databases/s2ds5/images"
S2DS_GT_DIR = "/home/skye/data/Skye/databases/s2ds5/labs"

# 推理参数 (必须与之前一致)
CROP_SIZE = 512
STRIDE = 256 
# ==========================================

def sliding_window_predict(model, img_tensor):
    b, c, h, w = img_tensor.shape
    full_mask = torch.zeros((1, 1, h, w)).cuda()
    count_mask = torch.zeros((1, 1, h, w)).cuda()

    for y in range(0, h, STRIDE):
        for x in range(0, w, STRIDE):
            y1 = min(y, h - CROP_SIZE) if h > CROP_SIZE else 0
            x1 = min(x, w - CROP_SIZE) if w > CROP_SIZE else 0
            y2 = y1 + CROP_SIZE
            x2 = x1 + CROP_SIZE
            patch = img_tensor[:, :, y1:y2, x1:x2]
            with torch.no_grad():
                preds = model(patch)
                patch_pred = torch.sigmoid(preds[-1]) 
            full_mask[:, :, y1:y2, x1:x2] += patch_pred
            count_mask[:, :, y1:y2, x1:x2] += 1
            if w <= CROP_SIZE: break
        if h <= CROP_SIZE: break
    return full_mask / count_mask

def main():
    # 1. 加载模型
    net = WPFormer(channel=64).cuda()
    if not os.path.exists(MODEL_PATH):
        print(f"❌ 未找到权重文件: {MODEL_PATH}")
        return
    net.load_state_dict(torch.load(MODEL_PATH))
    net.eval()
    print(f"✅ 成功加载合成域预训练模型: {MODEL_PATH}")

    # 2. 统计量初始化
    total_tp, total_fp, total_fn, total_union = 0, 0, 0, 0
    img_names = sorted(os.listdir(S2DS_IMG_DIR))
    
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    # 3. 开始遍历真实数据集
    for name in tqdm(img_names, desc="Evaluating Sim-to-Real"):
        # 读取原图
        img_path = os.path.join(S2DS_IMG_DIR, name)
        ori_img = cv2.imread(img_path)
        img_rgb = cv2.cvtColor(ori_img, cv2.COLOR_BGR2RGB)
        h, w, _ = img_rgb.shape

        # 预处理
        img_input = (img_rgb.astype(np.float32) / 255.0 - mean) / std
        img_tensor = torch.from_numpy(img_input).permute(2, 0, 1).unsqueeze(0).float().cuda()

        # 推理
        pred_map = sliding_window_predict(net, img_tensor)
        preds_bin = (pred_map > 0.5).float()

        # 读取并映射 GT
        gt_path = os.path.join(S2DS_GT_DIR, os.path.splitext(name)[0] + ".png")
        if os.path.exists(gt_path):
            gt_img = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
            # 严格对齐映射：255 是裂缝，1-254 是忽略
            gt_tensor = torch.from_numpy(gt_img).unsqueeze(0).unsqueeze(0).cuda().float()
            
            # 建立评估掩码
            valid_mask = (gt_tensor != 0) # 只有标注了的地方(包含255和病害)才进入计算？ 
            # 不，根据 S2DS 习惯，应该是排除 1-254
            final_valid_mask = (gt_tensor == 0) | (gt_tensor == 255)
            
            # 正样本是 255
            target = (gt_tensor == 255).float()
            
            # 只在有效区域计算
            p_v = preds_bin[final_valid_mask]
            t_v = target[final_valid_mask]

            total_tp += (p_v * t_v).sum().item()
            total_fp += (p_v * (1 - t_v)).sum().item()
            total_fn += ((1 - p_v) * t_v).sum().item()
            total_union += (p_v + t_v).gt(0).sum().item()

    # 4. 计算最终指标
    precision = total_tp / (total_tp + total_fp + 1e-8)
    recall = total_tp / (total_tp + total_fn + 1e-8)
    f1_score = 2 * precision * recall / (precision + recall + 1e-8)
    iou = total_tp / (total_union + 1e-8)

    print("\n" + "="*40)
    print("📊 [Sim-to-Real Direct Transfer Results]")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-Score:  {f1_score:.4f}")
    print(f"IoU:       {iou:.4f}")
    print("="*40)

if __name__ == "__main__":
    main()