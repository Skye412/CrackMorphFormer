# -*- coding: utf-8 -*-
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

# ================= 配置区 =================
IMG_PATH = "/home/skye/data/Skye/databases/s2ds5/images/s2ds_000.png"
GT_PATH = "/home/skye/data/Skye/databases/s2ds5/labs/s2ds_000.png"

OUT_DIR = "/home/skye/data/Skye/CrackMorphFormer/figures/sgmpp_cues_s2ds_000_v2"
SIZE = 256
SIGMA = 1.5

# 新增：Gamma校正函数，用于增强对比度
def adjust_gamma(image, gamma=1.0):
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(image, table)

def mkdir(path):
    os.makedirs(path, exist_ok=True)

def normalize01(x, eps=1e-8):
    x = x.astype(np.float32)
    return (x - x.min()) / (x.max() - x.min() + eps)

def read_image(path, size=256):
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(path)
    img = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    return rgb, gray

def read_gt(path, size=256):
    gt = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if gt is None:
        raise FileNotFoundError(path)

    if gt.ndim == 3:
        gt = cv2.cvtColor(gt, cv2.COLOR_BGR2GRAY)

    gt = cv2.resize(gt, (size, size), interpolation=cv2.INTER_NEAREST)
    mask = (gt > 0).astype(np.float32)
    return mask

def compute_structure_tensor(gray, sigma=1.5, eps=1e-6):
    sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
    sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float32)

    gx = cv2.filter2D(gray, -1, sobel_x, borderType=cv2.BORDER_REFLECT)
    gy = cv2.filter2D(gray, -1, sobel_y, borderType=cv2.BORDER_REFLECT)

    edge_strength = np.sqrt(gx ** 2 + gy ** 2 + eps)

    gx2 = gx * gx
    gy2 = gy * gy
    gxgy = gx * gy

    ksize = int(2 * sigma * 3 + 1)
    if ksize % 2 == 0:
        ksize += 1

    jxx = cv2.GaussianBlur(gx2, (ksize, ksize), sigmaX=sigma, sigmaY=sigma)
    jxy = cv2.GaussianBlur(gxgy, (ksize, ksize), sigmaX=sigma, sigmaY=sigma)
    jyy = cv2.GaussianBlur(gy2, (ksize, ksize), sigmaX=sigma, sigmaY=sigma)

    trace = jxx + jyy
    delta = ((jxx - jyy) * 0.5) ** 2 + jxy ** 2
    sqrt_delta = np.sqrt(np.maximum(delta, eps))

    lambda1 = trace * 0.5 + sqrt_delta
    lambda2 = trace * 0.5 - sqrt_delta

    denom = np.sqrt(jxy ** 2 + (lambda2 - jxx) ** 2 + eps)
    dir_x = -jxy / denom
    dir_y = (lambda2 - jxx) / denom

    tubularity = (lambda1 - lambda2) / (lambda1 + lambda2 + eps)
    tubularity = np.clip(tubularity, 0.0, 1.0)

    theta = np.arctan2(dir_y, dir_x)

    return {
        "gx": gx,
        "gy": gy,
        "edge_strength": edge_strength,
        "lambda1": lambda1,
        "lambda2": lambda2,
        "dir_x": dir_x,
        "dir_y": dir_y,
        "theta": theta,
        "tubularity": tubularity,
    }

def save_structure_tensor(gray, mask, data, out_path):
    # --- 修改 1: 增加箭头密度和尺寸 ---
    base = normalize01(gray)
    base_rgb = np.dstack([base, base, base])

    # 绘制 GT 轮廓作为背景参考
    contour = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8))
    base_rgb[contour > 0] = [1.0, 1.0, 1.0]

    h, w = gray.shape

    # 将步长从 16 减小到 8，密度增加 4 倍
    step = 8
    yy, xx = np.mgrid[step // 2:h:step, step // 2:w:step]

    valid = mask[yy, xx] > 0
    dx = data["dir_x"][yy, xx]
    dy = data["dir_y"][yy, xx]

    fig, ax = plt.subplots(figsize=(3, 3), dpi=250)
    ax.imshow(base_rgb)

    # 筛选有效点
    # 注意：quiver 的 scale 参数需要根据步长调整，步长变小，scale 需要相应调整以保持箭头长度合适
    q = ax.quiver(
        xx[valid],
        yy[valid],
        dx[valid],
        dy[valid],
        color="#ffcc00",  # 更亮一点的黄色
        angles="xy",
        scale_units="xy",
        scale=0.18,       # 调整比例以适应更密的网格
        width=0.012,      # 加宽箭杆
        headwidth=4.0,    # 加大箭头头部
        headlength=5.0,
        zorder=10         # 强制置于顶层
    )

    ax.set_axis_off()
    plt.tight_layout(pad=0)
    plt.savefig(out_path, bbox_inches="tight", pad_inches=0)
    plt.close()

def save_orientation(mask, data, out_path):
    theta = data["theta"]
    edge = normalize01(data["edge_strength"])

    hue = ((theta + np.pi) / (2 * np.pi) * 179).astype(np.uint8)
    sat = np.full_like(hue, 255, dtype=np.uint8)
    val = np.maximum((mask * 255), (edge * 130)).astype(np.uint8)
    hsv = np.dstack([hue, sat, val])
    rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

    bg = np.zeros_like(rgb)
    rgb = np.where(mask[..., None] > 0, rgb, (rgb * 0.25 + bg * 0.75).astype(np.uint8))

    plt.imsave(out_path, rgb)

def save_tubularity(mask, data, out_path):
    tub = normalize01(data["tubularity"])
    tub = tub * (0.20 + 0.80 * mask)
    tub = cv2.GaussianBlur(tub, (3, 3), 0)
    plt.imsave(out_path, tub, cmap="gray", vmin=0, vmax=1)

def save_edge_confidence(mask, data, out_path):
    # --- 修改 2: 强化边缘显示 ---
    edge = normalize01(data["edge_strength"])

    # 1. 移除高斯模糊，保留锐利边缘
    # edge = cv2.GaussianBlur(edge, (3, 3), 0) (已移除)

    # 2. 使用 Gamma 校正增强对比度 (gamma < 1 提亮高亮区，压暗背景)
    # 将 float [0,1] 转为 uint8 [0,255] 进行处理
    edge_uint8 = (edge * 255).astype(np.uint8)
    enhanced_edge = adjust_gamma(edge_uint8, gamma=0.6)

    # 3. 可选：再次轻微叠加 mask 确保 GT 区域绝对可见，但主要依赖 Sobel 强度
    # 这里我们直接保存增强后的 Sobel 结果，因为它更能反映 "Edge Confidence" 的本质
    plt.imsave(out_path, enhanced_edge, cmap="gray", vmin=0, vmax=255)

def save_preview(paths, out_path):
    titles = [
        "Structure Tensor",
        "Orientation",
        "Tubularity",
        "Edge Confidence",
    ]

    fig, axes = plt.subplots(1, 4, figsize=(8, 2.2), dpi=250)

    for ax, title, path in zip(axes, titles, paths):
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if img.ndim == 2:
            ax.imshow(img, cmap="gray")
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            ax.imshow(img)

        ax.set_title(title, fontsize=8)
        ax.set_axis_off()

    plt.tight_layout(w_pad=0.2)
    plt.savefig(out_path, bbox_inches="tight", pad_inches=0.05)
    plt.close()

def main():
    mkdir(OUT_DIR)

    rgb, gray = read_image(IMG_PATH, SIZE)
    mask = read_gt(GT_PATH, SIZE)

    data = compute_structure_tensor(gray, sigma=SIGMA)

    p1 = os.path.join(OUT_DIR, "01_structure_tensor.png")
    p2 = os.path.join(OUT_DIR, "02_orientation_theta.png")
    p3 = os.path.join(OUT_DIR, "03_tubularity_tau.png")
    p4 = os.path.join(OUT_DIR, "04_edge_confidence_e.png")
    p5 = os.path.join(OUT_DIR, "00_preview.png")

    save_structure_tensor(gray, mask, data, p1)
    save_orientation(mask, data, p2)
    save_tubularity(mask, data, p3)
    save_edge_confidence(mask, data, p4)
    save_preview([p1, p2, p3, p4], p5)

    print("Done. Saved to:")
    print(OUT_DIR)

if __name__ == "__main__":
    main()