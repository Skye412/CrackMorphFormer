# -*- coding: utf-8 -*-
import os
import random
import cv2
import torch
import numpy as np
from PIL import Image, ImageEnhance
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import albumentations as albu

class ImageFolder(Dataset):
    def __init__(self, image_root, gt_root, trainsize=384, is_train=False):
        self.training = is_train
        self.trainsize = trainsize

        self.images = [os.path.join(image_root, f) for f in os.listdir(image_root)]
        self.gts = [os.path.join(gt_root, p) for p in os.listdir(gt_root)]
        self.images = sorted(self.images)
        self.gts = sorted(self.gts)

        # 数据增强 (只保留像素级和翻转，移除会导致裂缝变形的 Resize)
        self.aug_transform = albu.Compose([
            albu.HorizontalFlip(p=0.5),
            albu.VerticalFlip(p=0.5),
            albu.RandomRotate90(p=0.5),
        ])

        self.img_transform = self.get_transform()
        # mask 只需要转为 Tensor，不需要正则化
        self.gt_transform = transforms.ToTensor()

    def get_crop_params(self, gt_np):
        """ 7:3 动态裁剪策略 """
        h, w = gt_np.shape
        crop_size = self.trainsize
        
        # 如果图像比 crop_size 小，直接返回左上角(0,0)，后续进行 padding 或 resize
        if h <= crop_size or w <= crop_size:
            return 0, 0

        # 70% 概率强行在裂缝周围裁剪 (前提是图里有裂缝，假设裂缝像素 > 0)
        if random.random() < 0.7 and np.any(gt_np > 0):
            y_indices, x_indices = np.where(gt_np > 0)
            idx = random.randint(0, len(y_indices) - 1)
            center_y, center_x = y_indices[idx], x_indices[idx]
            
            # 引入随机抖动，防止每次裂缝都在正中心
            jitter_y = random.randint(-crop_size//2, crop_size//2)
            jitter_x = random.randint(-crop_size//2, crop_size//2)
            
            y1 = max(0, center_y - crop_size // 2 + jitter_y)
            x1 = max(0, center_x - crop_size // 2 + jitter_x)
            
            # 确保不越界
            y1 = min(h - crop_size, y1)
            x1 = min(w - crop_size, x1)
        else:
            # 30% 概率纯随机裁剪（充当纯背景或包含噪点的难负样本）
            y1 = random.randint(0, h - crop_size)
            x1 = random.randint(0, w - crop_size)
            
        return y1, x1

    def __getitem__(self, index):
        image = cv2.imread(self.images[index])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        gt = cv2.imread(self.gts[index], cv2.IMREAD_GRAYSCALE)

        if self.training:
            # 使用 7:3 策略进行 Patch 裁剪
            y1, x1 = self.get_crop_params(gt)
            if gt.shape[0] > self.trainsize and gt.shape[1] > self.trainsize:
                image = image[y1:y1+self.trainsize, x1:x1+self.trainsize]
                gt = gt[y1:y1+self.trainsize, x1:x1+self.trainsize]
            else:
                # 兜底：如果原图很小，则 Resize
                image = cv2.resize(image, (self.trainsize, self.trainsize))
                gt = cv2.resize(gt, (self.trainsize, self.trainsize), interpolation=cv2.INTER_NEAREST)
            
            # 执行 Albumentations 数据增强
            augmented = self.aug_transform(image=image, mask=gt)
            image, gt = augmented['image'], augmented['mask']
        else:
            # 测试时为了对齐评估，保持原有的缩放逻辑（后续再引入滑窗推理）
            image = cv2.resize(image, (self.trainsize, self.trainsize))
            gt = cv2.resize(gt, (self.trainsize, self.trainsize), interpolation=cv2.INTER_NEAREST)

        image = Image.fromarray(image)
        gt = Image.fromarray(gt)

        image = self.img_transform(image)
        gt = self.gt_transform(gt)

        return {'image': image, 'label': gt}

    def get_transform(self, mean=None, std=None):
        mean = [0.485, 0.456, 0.406] if mean is None else mean
        std = [0.229, 0.224, 0.225] if std is None else std
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])

    def __len__(self):
        return len(self.images)

def get_loader(image_root, gt_root, batchsize, trainsize, is_train=False, shuffle=True, num_workers=0, pin_memory=True):
    dataset = ImageFolder(image_root, gt_root, trainsize, is_train)
    data_loader = DataLoader(dataset=dataset, batch_size=batchsize, shuffle=shuffle,
                             num_workers=num_workers, pin_memory=pin_memory, drop_last=True)
    return data_loader