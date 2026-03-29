import os
import torch
import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm
import torch.nn.functional as F
from model.WPFormer import WPFormer

# ================= 配置区 =================
# 根路径配置
BASE_DATA_DIR = "/home/skye/data/Skye/databases"
SAVE_ROOT = "vis_results"

# 模型权重路径映射
CHECKPOINTS = {
    "synth": "save/best_synth_pretrained.pth",
    "fold1": "save/best_finetune_fold1.pth",
    "fold2": "save/best_finetune_fold2.pth",
    "fold3": "save/best_finetune_fold3.pth",
    "fold4": "save/best_finetune_fold4.pth",
    "fold5": "save/best_finetune_fold5.pth",
}

# 推理参数
CROP_SIZE = 512
STRIDE = 256  # 重叠 50% 保证平滑
# ==========================================

def sliding_window_predict(model, img_tensor, crop_size=512, stride=256):
    """
    针对大图的滑窗推理，带重叠区域平均平滑处理
    """
    b, c, h, w = img_tensor.shape
    full_mask = torch.zeros((1, 1, h, w)).cuda()
    count_mask = torch.zeros((1, 1, h, w)).cuda()

    for y in range(0, h, stride):
        for x in range(0, w, stride):
            # 边界处理：如果最后一块超出，则往回缩
            y1 = min(y, h - crop_size) if h > crop_size else 0
            x1 = min(x, w - crop_size) if w > crop_size else 0
            y2 = y1 + crop_size
            x2 = x1 + crop_size

            patch = img_tensor[:, :, y1:y2, x1:x2]
            
            with torch.no_grad():
                preds = model(patch)
                # 取最后一个最高分辨率的输出
                patch_pred = torch.sigmoid(preds[-1]) 

            full_mask[:, :, y1:y2, x1:x2] += patch_pred
            count_mask[:, :, y1:y2, x1:x2] += 1
            
            if w <= crop_size: break
        if h <= crop_size: break

    return (full_mask / count_mask).squeeze().cpu().numpy()

def process_images(model, img_dir, gt_dir, img_list, save_dir):
    """
    核心处理函数
    """
    os.makedirs(save_dir, exist_ok=True)
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    for name in tqdm(img_list, desc=f"Saving to {save_dir}", leave=False):
        # 1. 读取原图
        img_path = os.path.join(img_dir, name)
        if not os.path.exists(img_path): continue
        ori_img = cv2.imread(img_path)
        img_rgb = cv2.cvtColor(ori_img, cv2.COLOR_BGR2RGB)
        h, w, _ = img_rgb.shape

        # 2. 预处理
        img_input = img_rgb.astype(np.float32) / 255.0
        img_input = (img_input - mean) / std
        img_tensor = torch.from_numpy(img_input).permute(2, 0, 1).unsqueeze(0).float().cuda()

        # 3. 推理
        pred_map = sliding_window_predict(model, img_tensor, CROP_SIZE, STRIDE)

        # 4. 读取 GT (S2DS 的 label 文件夹是 labs, Synth 是 gt)
        # 统一处理文件名映射
        gt_name = os.path.splitext(name)[0] + ".png"
        gt_path = os.path.join(gt_dir, gt_name)
        if os.path.exists(gt_path):
            gt_img = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
            # 严格对齐映射：255是裂缝，1-254是其他(变黑)
            gt_vis = np.zeros_like(gt_img)
            gt_vis[gt_img == 255] = 255
        else:
            gt_vis = np.zeros((h, w), dtype=np.uint8)

        # 5. 后处理与染色
        pred_vis_gray = (pred_map * 255).astype(np.uint8)
        # 伪彩色热力图：显示置信度，红色代表模型非常确定是裂缝
        pred_heatmap = cv2.applyColorMap(pred_vis_gray, cv2.COLORMAP_JET)
        
        # 二值化结果
        _, pred_binary = cv2.threshold(pred_vis_gray, 127, 255, cv2.THRESH_BINARY)
        pred_binary_bgr = cv2.cvtColor(pred_binary, cv2.COLOR_GRAY2BGR)
        gt_bgr = cv2.cvtColor(gt_vis, cv2.COLOR_GRAY2BGR)

        # 6. 拼接展示 (为了方便查看，将超大图 Resize 到高度 800)
        show_h = 800
        show_w = int(800 * (w / h))
        vis_row = np.hstack([
            cv2.resize(ori_img, (show_w, show_h)),
            cv2.resize(gt_bgr, (show_w, show_h)),
            cv2.resize(pred_heatmap, (show_w, show_h)),
            cv2.resize(pred_binary_bgr, (show_w, show_h))
        ])

        # 保存结果
        cv2.imwrite(os.path.join(save_dir, name), vis_row)

def main():
    # 初始化模型
    net = WPFormer(channel=64).cuda()
    
    # ------------------ 1. 测试合成数据集 ------------------
    if os.path.exists(CHECKPOINTS["synth"]):
        print("\n--- Testing Synthetic Dataset ---")
        net.load_state_dict(torch.load(CHECKPOINTS["synth"]))
        net.eval()
        
        synth_img_dir = os.path.join(BASE_DATA_DIR, "synthcrack/images")
        synth_gt_dir = os.path.join(BASE_DATA_DIR, "synthcrack/gt")
        
        # 获取 500 张验证集 (逻辑与 Dataloader 一致)
        all_imgs = sorted(os.listdir(synth_img_dir))
        import random
        random.seed(42)
        random.shuffle(all_imgs)
        val_imgs = all_imgs[:500]
        
        process_images(net, synth_img_dir, synth_gt_dir, val_imgs, os.path.join(SAVE_ROOT, "synth"))
    
    # ------------------ 2. 测试 S2DS 五折数据 ------------------
    for f in range(1, 6):
        fold_key = f"fold{f}"
        if os.path.exists(CHECKPOINTS[fold_key]):
            print(f"\n--- Testing S2DS {fold_key.upper()} ---")
            net.load_state_dict(torch.load(CHECKPOINTS[fold_key]))
            net.eval()
            
            s2ds_img_dir = os.path.join(BASE_DATA_DIR, "s2ds5/images")
            s2ds_gt_dir = os.path.join(BASE_DATA_DIR, "s2ds5/labs")
            
            # 从对应的 txt 文件读取验证集图片名
            txt_path = os.path.join(BASE_DATA_DIR, f"s2ds5/fold{f}_val.txt")
            with open(txt_path, 'r') as file:
                val_imgs = [line.strip() for line in file.readlines()]
            
            process_images(net, s2ds_img_dir, s2ds_gt_dir, val_imgs, os.path.join(SAVE_ROOT, fold_key))
        else:
            print(f"⚠️ Skip {fold_key}: weight not found.")

    print(f"\n✅ All predictions done! Results saved in '{SAVE_ROOT}' folder.")

if __name__ == "__main__":
    main()