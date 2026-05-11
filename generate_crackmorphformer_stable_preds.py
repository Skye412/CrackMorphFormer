#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Generate S2DS prediction visualizations for CrackMorphFormer stable checkpoints.

Outputs under:
  <save-root>/<experiment>/
    pred_png/fold*/xxx.png
    prob_png/fold*/xxx.png
    overlay_png/fold*/xxx.png
    compare_png/fold*/xxx.png

The S2DS label mapping matches ESDI_dataloader.py:
  RGB white (255,255,255) -> crack foreground
  all other colors        -> background
"""

from __future__ import annotations

import argparse
import os
from typing import Any, Dict, List, Tuple

import numpy as np
from PIL import Image, ImageDraw

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from model.CrackMorphFormer import CrackMorphFormer


EXPERIMENTS: Dict[str, Dict[str, bool]] = {
    "synth2real_full": {"use_dfe": True, "use_sgmpp": True},
    "synth2real_wo_dfe": {"use_dfe": False, "use_sgmpp": True},
    "synth2real_wo_sgmpp": {"use_dfe": True, "use_sgmpp": False},
    "synth2real_wo_dfe_sgmpp": {"use_dfe": False, "use_sgmpp": False},
}


class S2DSPredDataset(Dataset):
    def __init__(self, data_root: str, fold: int, split: str = "val", train_size: int = 512):
        self.data_root = data_root
        self.fold = fold
        self.split = split
        self.train_size = train_size

        img_dir = os.path.join(data_root, "images")
        gt_dir = os.path.join(data_root, "labs")
        split_file = os.path.join(data_root, f"fold{fold}_{split}.txt")

        if not os.path.isfile(split_file):
            raise FileNotFoundError(f"Split file not found: {split_file}")

        self.samples: List[Tuple[str, str, str]] = []

        with open(split_file, "r", encoding="utf-8") as f:
            for line in f:
                name = line.strip()
                if not name:
                    continue

                img_path = os.path.join(img_dir, name)
                gt_path = os.path.join(gt_dir, os.path.splitext(name)[0] + ".png")

                if not os.path.isfile(img_path):
                    raise FileNotFoundError(f"Image not found: {img_path}")
                if not os.path.isfile(gt_path):
                    raise FileNotFoundError(f"GT not found: {gt_path}")

                self.samples.append((img_path, gt_path, os.path.basename(name)))

        self.img_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

    @staticmethod
    def pad_if_needed(image: np.ndarray, target: np.ndarray, size: int) -> Tuple[np.ndarray, np.ndarray]:
        h, w = target.shape[:2]
        pad_h = max(0, size - h)
        pad_w = max(0, size - w)

        if pad_h == 0 and pad_w == 0:
            return image, target

        image = np.pad(
            image,
            ((0, pad_h), (0, pad_w), (0, 0)),
            mode="reflect",
        )
        target = np.pad(
            target,
            ((0, pad_h), (0, pad_w)),
            mode="constant",
            constant_values=0,
        )
        return image, target

    @staticmethod
    def center_crop_np(arr: np.ndarray, size: int) -> np.ndarray:
        h, w = arr.shape[:2]
        top = max(0, (h - size) // 2)
        left = max(0, (w - size) // 2)
        return arr[top:top + size, left:left + size]

    @staticmethod
    def map_s2ds_rgb(gt_rgb: np.ndarray) -> np.ndarray:
        crack = np.all(gt_rgb == np.array([255, 255, 255], dtype=np.uint8), axis=-1)
        return crack.astype(np.uint8)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        img_path, gt_path, name = self.samples[index]

        image_rgb = np.asarray(Image.open(img_path).convert("RGB"))
        gt_rgb = np.asarray(Image.open(gt_path).convert("RGB"))
        target = self.map_s2ds_rgb(gt_rgb)

        image_rgb, target = self.pad_if_needed(image_rgb, target, self.train_size)
        image_rgb = self.center_crop_np(image_rgb, self.train_size)
        target = self.center_crop_np(target, self.train_size)

        return {
            "image": self.img_transform(Image.fromarray(image_rgb)),
            "image_rgb": torch.from_numpy(image_rgb.copy()).permute(2, 0, 1),
            "label": torch.from_numpy(target).unsqueeze(0),
            "name": name,
        }


def recursive_last_tensor(x: Any) -> torch.Tensor:
    if torch.is_tensor(x):
        return x

    if isinstance(x, (list, tuple)):
        for item in reversed(x):
            try:
                return recursive_last_tensor(item)
            except TypeError:
                pass

    if isinstance(x, dict):
        for key in ["pred", "preds", "out", "output", "logits", "mask", "masks"]:
            if key in x:
                try:
                    return recursive_last_tensor(x[key])
                except TypeError:
                    pass

        for item in reversed(list(x.values())):
            try:
                return recursive_last_tensor(item)
            except TypeError:
                pass

    raise TypeError(f"Could not find tensor in output type: {type(x)}")


def load_model(args, exp_name: str, ckpt_path: str) -> torch.nn.Module:
    cfg = EXPERIMENTS[exp_name]

    model = CrackMorphFormer(
        channel=args.channel,
        num_queries=args.num_queries,
        backbone_path=args.backbone_path,
        use_dfe=cfg["use_dfe"],
        use_sgmpp=cfg["use_sgmpp"],
    )

    state = torch.load(ckpt_path, map_location="cpu")
    if isinstance(state, dict) and "model_state" in state:
        state = state["model_state"]

    msg = model.load_state_dict(state, strict=True)
    missing = list(getattr(msg, "missing_keys", []))
    unexpected = list(getattr(msg, "unexpected_keys", []))
    if missing or unexpected:
        raise RuntimeError(
            f"Unexpected checkpoint load result: missing={len(missing)}, "
            f"unexpected={len(unexpected)}"
        )

    return model.to(args.device).eval()


def colorize_gt(gt: np.ndarray) -> np.ndarray:
    out = np.zeros((gt.shape[0], gt.shape[1], 3), dtype=np.uint8)
    out[gt == 1] = [255, 255, 255]
    return out


def make_overlay(image_rgb: np.ndarray, pred_bin: np.ndarray, alpha: float = 0.55) -> np.ndarray:
    overlay = image_rgb.astype(np.float32).copy()
    red = np.zeros_like(overlay)
    red[..., 0] = 255

    mask = pred_bin.astype(bool)
    overlay[mask] = (1.0 - alpha) * overlay[mask] + alpha * red[mask]

    return np.clip(overlay, 0, 255).astype(np.uint8)


def add_title(img: Image.Image, title: str, height: int = 28) -> Image.Image:
    canvas = Image.new("RGB", (img.width, img.height + height), "white")
    canvas.paste(img, (0, height))

    draw = ImageDraw.Draw(canvas)
    draw.text((8, 6), title, fill=(0, 0, 0))

    return canvas


def save_compare(
    image_rgb: np.ndarray,
    gt: np.ndarray,
    pred_bin: np.ndarray,
    overlay: np.ndarray,
    save_path: str,
):
    gt_rgb = colorize_gt(gt)
    pred_rgb = np.repeat((pred_bin * 255).astype(np.uint8)[..., None], 3, axis=2)

    panels = [
        add_title(Image.fromarray(image_rgb), "Image"),
        add_title(Image.fromarray(gt_rgb), "GT"),
        add_title(Image.fromarray(pred_rgb), "Pred"),
        add_title(Image.fromarray(overlay), "Overlay"),
    ]

    w = sum(p.width for p in panels)
    h = max(p.height for p in panels)
    canvas = Image.new("RGB", (w, h), "white")

    x = 0
    for panel in panels:
        canvas.paste(panel, (x, 0))
        x += panel.width

    canvas.save(save_path)


def ensure_dirs(save_root: str, exp_name: str, fold: int) -> Dict[str, str]:
    exp_root = os.path.join(save_root, exp_name)
    dirs = {
        "pred": os.path.join(exp_root, "pred_png", f"fold{fold}"),
        "prob": os.path.join(exp_root, "prob_png", f"fold{fold}"),
        "overlay": os.path.join(exp_root, "overlay_png", f"fold{fold}"),
        "compare": os.path.join(exp_root, "compare_png", f"fold{fold}"),
    }

    for directory in dirs.values():
        os.makedirs(directory, exist_ok=True)

    return dirs


def checkpoint_path(args, exp_name: str, fold: int) -> str:
    return os.path.join(
        args.exp_root,
        exp_name,
        "finetune",
        "weights",
        f"best_finetune_fold{fold}.pth",
    )


def generate_one_fold(exp_name: str, fold: int, args) -> int:
    ckpt_path = checkpoint_path(args, exp_name, fold)
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    print(f"\n{'=' * 88}")
    print(f"Experiment: {exp_name} | Fold {fold}")
    print(f"Checkpoint: {ckpt_path}")
    print(f"{'=' * 88}")

    dataset = S2DSPredDataset(
        data_root=args.s2ds_root,
        fold=fold,
        split=args.split,
        train_size=args.train_size,
    )

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=str(args.device).startswith("cuda"),
    )

    model = load_model(args, exp_name, ckpt_path)
    dirs = ensure_dirs(args.save_root, exp_name, fold)

    count = 0

    with torch.no_grad():
        for batch in loader:
            image = batch["image"].to(args.device, non_blocking=True)
            image_rgb_batch = batch["image_rgb"].numpy()
            target = batch["label"].numpy()
            names = batch["name"]

            output = model(image)
            logits = recursive_last_tensor(output)

            if logits.ndim == 3:
                logits = logits.unsqueeze(1)

            if logits.shape[-2:] != target.shape[-2:]:
                logits = F.interpolate(
                    logits,
                    size=target.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )

            prob = torch.sigmoid(logits).detach().cpu().numpy()
            pred = (prob >= args.threshold).astype(np.uint8)

            for i in range(pred.shape[0]):
                name = names[i]
                stem = os.path.splitext(name)[0]

                image_rgb = image_rgb_batch[i].transpose(1, 2, 0).astype(np.uint8)
                gt = target[i, 0].astype(np.uint8)
                prob_i = prob[i, 0]
                pred_i = pred[i, 0].astype(np.uint8)

                prob_u8 = np.clip(prob_i * 255.0, 0, 255).astype(np.uint8)
                pred_u8 = (pred_i * 255).astype(np.uint8)
                overlay = make_overlay(image_rgb, pred_i, alpha=args.overlay_alpha)

                Image.fromarray(pred_u8).save(os.path.join(dirs["pred"], f"{stem}.png"))
                Image.fromarray(prob_u8).save(os.path.join(dirs["prob"], f"{stem}.png"))
                Image.fromarray(overlay).save(os.path.join(dirs["overlay"], f"{stem}.png"))
                save_compare(
                    image_rgb=image_rgb,
                    gt=gt,
                    pred_bin=pred_i,
                    overlay=overlay,
                    save_path=os.path.join(dirs["compare"], f"{stem}.png"),
                )

                count += 1

    print(f"Saved {count} samples to: {os.path.abspath(os.path.join(args.save_root, exp_name))}")

    del model
    if torch.cuda.is_available() and str(args.device).startswith("cuda"):
        torch.cuda.empty_cache()

    return count


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--exp-root",
        default="/home/skye/data/Skye/CrackMorphFormer/results/CrackMorphFormer_final_stable",
    )
    parser.add_argument(
        "--s2ds-root",
        default="/home/skye/data/Skye/databases/s2ds5",
    )
    parser.add_argument(
        "--save-root",
        default="/home/skye/data/Skye/CrackMorphFormer/results/CrackMorphFormer_final_stable/predictions_s2ds5",
    )
    parser.add_argument(
        "--experiments",
        nargs="+",
        default=["synth2real_full"],
        choices=list(EXPERIMENTS.keys()),
    )
    parser.add_argument("--folds", nargs="+", type=int, default=[1, 2, 3, 4, 5])
    parser.add_argument("--split", default="val", choices=["train", "val"])
    parser.add_argument("--train-size", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--overlay-alpha", type=float, default=0.55)
    parser.add_argument("--channel", type=int, default=64)
    parser.add_argument("--num-queries", type=int, default=16)
    parser.add_argument("--backbone-path", default=None)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")

    return parser.parse_args()


def main():
    args = parse_args()

    total = 0
    for exp_name in args.experiments:
        for fold in args.folds:
            total += generate_one_fold(exp_name, fold, args)

    print("\nDone.")
    print(f"Saved {total} prediction sets.")
    print(f"Prediction root: {os.path.abspath(args.save_root)}")


if __name__ == "__main__":
    main()
