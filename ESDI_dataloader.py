# -*- coding: utf-8 -*-
import os
import random
from typing import Tuple

import cv2
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import albumentations as albu


# ============================================================
# S2DS RGB label definitions
# ============================================================
S2DS_CRACK = np.array([255, 255, 255], dtype=np.uint8)
S2DS_BACKGROUND = np.array([0, 0, 0], dtype=np.uint8)

# Other defect colors in S2DS.
# A-mainline policy:
#   crack white            -> foreground 1
#   black background       -> background 0
#   all other defect colors -> background 0
S2DS_SPALLING = np.array([255, 0, 0], dtype=np.uint8)
S2DS_CORROSION = np.array([255, 255, 0], dtype=np.uint8)
S2DS_EFFLORESCENCE = np.array([0, 255, 255], dtype=np.uint8)
S2DS_VEGETATION = np.array([0, 255, 0], dtype=np.uint8)
S2DS_CONTROL_POINT = np.array([0, 0, 255], dtype=np.uint8)


def _safe_read_rgb(path: str) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"))


def _safe_read_gray(path: str) -> np.ndarray:
    return np.asarray(Image.open(path).convert("L"))


def _pad_if_needed(
    image: np.ndarray,
    target: np.ndarray,
    crack_mask: np.ndarray,
    crop_size: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    h, w = target.shape[:2]

    pad_h = max(0, crop_size - h)
    pad_w = max(0, crop_size - w)

    if pad_h == 0 and pad_w == 0:
        return image, target, crack_mask

    image = cv2.copyMakeBorder(
        image,
        0,
        pad_h,
        0,
        pad_w,
        borderType=cv2.BORDER_REFLECT_101,
    )

    target = cv2.copyMakeBorder(
        target,
        0,
        pad_h,
        0,
        pad_w,
        borderType=cv2.BORDER_CONSTANT,
        value=0,
    )

    crack_mask = cv2.copyMakeBorder(
        crack_mask,
        0,
        pad_h,
        0,
        pad_w,
        borderType=cv2.BORDER_CONSTANT,
        value=0,
    )

    return image, target, crack_mask


def _random_crop(
    image: np.ndarray,
    target: np.ndarray,
    crop_size: int,
) -> Tuple[np.ndarray, np.ndarray]:
    h, w = target.shape[:2]

    top = random.randint(0, max(0, h - crop_size))
    left = random.randint(0, max(0, w - crop_size))

    image = image[top:top + crop_size, left:left + crop_size]
    target = target[top:top + crop_size, left:left + crop_size]

    return image, target


def _crack_aware_crop(
    image: np.ndarray,
    target: np.ndarray,
    crack_mask: np.ndarray,
    crop_size: int,
    dilation_kernel: int = 21,
) -> Tuple[np.ndarray, np.ndarray]:
    h, w = crack_mask.shape[:2]

    if crack_mask.sum() <= 0:
        return _random_crop(image, target, crop_size)

    kernel = np.ones((dilation_kernel, dilation_kernel), dtype=np.uint8)
    dilated = cv2.dilate(crack_mask.astype(np.uint8), kernel, iterations=1)

    ys, xs = np.where(dilated > 0)

    if len(xs) == 0:
        return _random_crop(image, target, crop_size)

    idx = random.randint(0, len(xs) - 1)
    cy = int(ys[idx])
    cx = int(xs[idx])

    top = int(np.clip(cy - crop_size // 2, 0, h - crop_size))
    left = int(np.clip(cx - crop_size // 2, 0, w - crop_size))

    image = image[top:top + crop_size, left:left + crop_size]
    target = target[top:top + crop_size, left:left + crop_size]

    return image, target


def _center_crop(
    image: np.ndarray,
    target: np.ndarray,
    crop_size: int,
) -> Tuple[np.ndarray, np.ndarray]:
    h, w = target.shape[:2]

    top = max(0, (h - crop_size) // 2)
    left = max(0, (w - crop_size) // 2)

    image = image[top:top + crop_size, left:left + crop_size]
    target = target[top:top + crop_size, left:left + crop_size]

    return image, target


class CrackDataset(Dataset):
    """
    A-mainline dataset loader.

    S2DS binary crack setting:
      - crack label:       RGB (255, 255, 255) -> 1
      - background:        RGB (0, 0, 0)       -> 0
      - other defects:     all non-white colors -> 0

    This matches the final A_bg_no_cldice setting:
      - other defects are treated as background
      - no ignore label is used for S2DS
      - crop is guided only by crack pixels
    """

    def __init__(
        self,
        data_root: str,
        dataset_name: str = "s2ds5",
        mode: str = "train",
        fold: int = 1,
        trainsize: int = 512,
        pos_crop_prob: float = 0.75,
        dilation_kernel: int = 21,
    ):
        super().__init__()

        self.data_root = data_root
        self.dataset_name = dataset_name
        self.mode = mode
        self.fold = fold
        self.trainsize = trainsize
        self.pos_crop_prob = pos_crop_prob
        self.dilation_kernel = dilation_kernel

        if self.mode not in ["train", "val", "test"]:
            raise ValueError(f"Unsupported mode: {self.mode}")

        self.images = []
        self.gts = []

        self.aug_transform = albu.Compose([
            albu.HorizontalFlip(p=0.5),
            albu.VerticalFlip(p=0.5),
            albu.RandomRotate90(p=0.5),
        ])

        self.img_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

        if self.dataset_name == "s2ds5":
            self._load_s2ds5()
        elif self.dataset_name == "synthcrack":
            self._load_synthcrack()
        else:
            raise ValueError(f"Unsupported dataset_name: {self.dataset_name}")

        if len(self.images) == 0:
            raise RuntimeError(
                f"No samples found: dataset={self.dataset_name}, "
                f"mode={self.mode}, fold={self.fold}, root={self.data_root}"
            )

    def _load_s2ds5(self):
        img_dir = os.path.join(self.data_root, "images")
        gt_dir = os.path.join(self.data_root, "labs")
        txt_file = os.path.join(self.data_root, f"fold{self.fold}_{self.mode}.txt")

        if not os.path.isfile(txt_file):
            raise FileNotFoundError(f"Split file not found: {txt_file}")

        if not os.path.isdir(img_dir):
            raise FileNotFoundError(f"Image directory not found: {img_dir}")

        if not os.path.isdir(gt_dir):
            raise FileNotFoundError(f"Label directory not found: {gt_dir}")

        with open(txt_file, "r", encoding="utf-8") as f:
            lines = f.readlines()

        for line in lines:
            name = line.strip()
            if not name:
                continue

            img_path = os.path.join(img_dir, name)
            gt_name = os.path.splitext(name)[0] + ".png"
            gt_path = os.path.join(gt_dir, gt_name)

            if not os.path.isfile(img_path):
                raise FileNotFoundError(f"Image not found: {img_path}")

            if not os.path.isfile(gt_path):
                raise FileNotFoundError(f"Label not found: {gt_path}")

            self.images.append(img_path)
            self.gts.append(gt_path)

    def _load_synthcrack(self):
        img_dir = os.path.join(self.data_root, "images")
        gt_dir = os.path.join(self.data_root, "gt")

        if not os.path.isdir(img_dir):
            raise FileNotFoundError(f"Image directory not found: {img_dir}")

        if not os.path.isdir(gt_dir):
            raise FileNotFoundError(f"Label directory not found: {gt_dir}")

        all_imgs = sorted([
            x for x in os.listdir(img_dir)
            if x.lower().endswith((".png", ".jpg", ".jpeg", ".bmp"))
        ])

        random.seed(42)
        random.shuffle(all_imgs)

        # 500 validation images, remaining images for training.
        val_imgs = all_imgs[:500]
        train_imgs = all_imgs[500:]

        if self.mode == "train":
            target_list = train_imgs
        else:
            target_list = val_imgs

        for name in target_list:
            img_path = os.path.join(img_dir, name)
            gt_name = os.path.splitext(name)[0] + ".png"
            gt_path = os.path.join(gt_dir, gt_name)

            if not os.path.isfile(img_path):
                raise FileNotFoundError(f"Image not found: {img_path}")

            if not os.path.isfile(gt_path):
                raise FileNotFoundError(f"Label not found: {gt_path}")

            self.images.append(img_path)
            self.gts.append(gt_path)

    def _map_s2ds_rgb(self, gt_rgb: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        RGB precise mapping for S2DS.

        A-mainline policy:
          white crack pixels -> 1
          all other pixels   -> 0
        """
        crack_mask = np.all(gt_rgb == S2DS_CRACK, axis=-1).astype(np.uint8)
        target = crack_mask.astype(np.float32)

        return target, crack_mask

    def _map_synth_label(self, gt_gray: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Synthetic crack labels are binary masks.

        white crack pixels -> 1
        others             -> 0
        """
        crack_mask = (gt_gray == 255).astype(np.uint8)
        target = crack_mask.astype(np.float32)

        return target, crack_mask

    def __getitem__(self, index: int):
        image = _safe_read_rgb(self.images[index])

        if self.dataset_name == "s2ds5":
            gt_rgb = _safe_read_rgb(self.gts[index])
            target, crack_mask = self._map_s2ds_rgb(gt_rgb)
        else:
            gt_gray = _safe_read_gray(self.gts[index])
            target, crack_mask = self._map_synth_label(gt_gray)

        image, target, crack_mask = _pad_if_needed(
            image=image,
            target=target,
            crack_mask=crack_mask,
            crop_size=self.trainsize,
        )

        if self.mode == "train":
            if crack_mask.sum() > 0 and random.random() < self.pos_crop_prob:
                image, target = _crack_aware_crop(
                    image=image,
                    target=target,
                    crack_mask=crack_mask,
                    crop_size=self.trainsize,
                    dilation_kernel=self.dilation_kernel,
                )
            else:
                image, target = _random_crop(
                    image=image,
                    target=target,
                    crop_size=self.trainsize,
                )

            aug = self.aug_transform(image=image, mask=target)
            image = aug["image"]
            target = aug["mask"]

        else:
            image, target = _center_crop(
                image=image,
                target=target,
                crop_size=self.trainsize,
            )

        target = target.astype(np.float32)

        return {
            "image": self.img_transform(Image.fromarray(image)),
            "label": torch.from_numpy(target).unsqueeze(0),
            "name": os.path.basename(self.images[index]),
        }

    def __len__(self) -> int:
        return len(self.images)


def get_loader(
    data_root: str,
    dataset_name: str,
    mode: str,
    fold: int,
    batchsize: int,
    trainsize: int,
    shuffle: bool = True,
    num_workers: int = 4,
    pin_memory: bool = True,
    pos_crop_prob: float = 0.75,
    dilation_kernel: int = 21,
):
    dataset = CrackDataset(
        data_root=data_root,
        dataset_name=dataset_name,
        mode=mode,
        fold=fold,
        trainsize=trainsize,
        pos_crop_prob=pos_crop_prob,
        dilation_kernel=dilation_kernel,
    )

    loader = DataLoader(
        dataset=dataset,
        batch_size=batchsize,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=(mode == "train"),
    )

    return loader