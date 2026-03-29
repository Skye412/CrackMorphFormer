import cv2
import numpy as np
import os

# 替换为你服务器上的实际路径
gt_path = "/home/skye/data/Skye/databases/synthcrack/gt/"
first_gt = os.path.join(gt_path, os.listdir(gt_path)[0])

gt = cv2.imread(first_gt, cv2.IMREAD_GRAYSCALE)
unique_values = np.unique(gt)

print(f"检查文件: {first_gt}")
print(f"标签图中的唯一像素值: {unique_values}")

if 255 in unique_values:
    print("✅ 结论：裂缝像素值是 255，ToTensor() 会将其转为 1.0，无需修改代码。")
elif 1 in unique_values:
    print("❌ 警告：裂缝像素值是 1！ToTensor() 会将其转为 0.0039。")
    print("👉 你需要修改 DataLoader，将 gt 乘以 255，或者在 ToTensor 前处理。")