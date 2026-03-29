# -*- coding: utf-8 -*-
import os
import random
import cv2
import torch
import numpy as np
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import albumentations as albu

class CrackDataset(Dataset):
    def __init__(self, data_root, dataset_name='s2ds5', mode='train', fold=1, trainsize=512):
        self.mode = mode
        self.trainsize = trainsize
        self.dataset_name = dataset_name
        self.images = []
        self.gts = []

        # 强大的 Albumentations 数据增强
        self.aug_transform = albu.Compose([
            albu.HorizontalFlip(p=0.5),
            albu.VerticalFlip(p=0.5),
            albu.RandomRotate90(p=0.5),
        ])
        
        self.img_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        # ================= 路径加载逻辑 =================
        if dataset_name == 's2ds5':
            img_dir = os.path.join(data_root, 'images')
            gt_dir = os.path.join(data_root, 'labs') 
            txt_file = os.path.join(data_root, f'fold{fold}_{mode}.txt')
            
            with open(txt_file, 'r') as f:
                lines = f.readlines()
            for line in lines:
                name = line.strip()
                self.images.append(os.path.join(img_dir, name))
                self.gts.append(os.path.join(gt_dir, os.path.splitext(name)[0] + '.png'))

        elif dataset_name == 'synthcrack':
            img_dir = os.path.join(data_root, 'images')
            gt_dir = os.path.join(data_root, 'gt')
            all_imgs = sorted(os.listdir(img_dir))
            
            random.seed(42)
            random.shuffle(all_imgs)
            val_imgs = all_imgs[:500]
            train_imgs = all_imgs[500:]

            target_list = train_imgs if mode == 'train' else val_imgs
            for name in target_list:
                self.images.append(os.path.join(img_dir, name))
                self.gts.append(os.path.join(gt_dir, os.path.splitext(name)[0] + '.png'))

    def __getitem__(self, index):
        image = np.asarray(Image.open(self.images[index]).convert('RGB'))
        gt = np.asarray(Image.open(self.gts[index]).convert('L'))

        # ================= 裁剪与增强 =================
        if self.mode == 'train':
            # 使用高概率强制裁剪包含裂缝的区域，对抗极度不平衡
            cropper = albu.CropNonEmptyMaskIfExists(height=self.trainsize, width=self.trainsize, p=1.0)
            aug = cropper(image=image, mask=gt)
            image, gt_aug = self.aug_transform(image=aug['image'], mask=aug['mask']).values()
        else:
            # 验证集使用中心裁剪保证评估稳定
            aug = albu.CenterCrop(height=self.trainsize, width=self.trainsize, p=1.0)(image=image, mask=gt)
            image, gt_aug = aug['image'], aug['mask']

        # ================= 标签严格映射 (核心修复) =================
        gt_np = np.asarray(gt_aug, dtype=np.float32)
        target = np.zeros_like(gt_np)
        
        if self.dataset_name == 's2ds5':
            target[gt_np == 255] = 1.0           # 裂缝(255) -> 正样本(1.0)
            target[(gt_np > 0) & (gt_np < 255)] = 255.0 # 剥落/泛碱等其他病害 -> 忽略(255.0)
        else:
            target[gt_np == 255] = 1.0           # 合成数据中完美的裂缝
            
        return {
            'image': self.img_transform(Image.fromarray(image)), 
            'label': torch.from_numpy(target).unsqueeze(0)
        }

    def __len__(self):
        return len(self.images)

def get_loader(data_root, dataset_name, mode, fold, batchsize, trainsize, shuffle=True, num_workers=4, pin_memory=True):
    dataset = CrackDataset(data_root, dataset_name, mode, fold, trainsize)
    return DataLoader(dataset=dataset, batch_size=batchsize, shuffle=shuffle,
                      num_workers=num_workers, pin_memory=pin_memory, drop_last=(mode=='train'))