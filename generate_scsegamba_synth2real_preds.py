#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Generate prediction maps for scsegamba_synth2real on S2DS 5-fold.

Outputs:
1. pred_png/fold*/xxx.png        binary prediction, 0/255
2. prob_png/fold*/xxx.png        probability map, 0~255
3. overlay_png/fold*/xxx.png     red prediction overlay on image
4. compare_png/fold*/xxx.png     image | GT | pred | overlay
"""

from __future__ import annotations

import argparse
import contextlib
import os
import sys
import types
from typing import Any, Dict, List, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


SCSegamba_SYNTH2REAL_CKPTS = {
    1: "/home/skye/data/Skye/third_party/SCSegamba/results/SCSegamba_Baseline_Sequential/synth2real/fold1/best.pth",
    2: "/home/skye/data/Skye/third_party/SCSegamba/results/SCSegamba_Baseline_Sequential/synth2real/fold2/best.pth",
    3: "/home/skye/data/Skye/third_party/SCSegamba/results/SCSegamba_Baseline_Sequential/synth2real/fold3/best.pth",
    4: "/home/skye/data/Skye/third_party/SCSegamba/results/SCSegamba_Baseline_Sequential/synth2real/fold4/best.pth",
    5: "/home/skye/data/Skye/third_party/SCSegamba/results/SCSegamba_Baseline_Sequential/synth2real/fold5/best.pth",
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
    def center_crop_np(arr: np.ndarray, size: int) -> np.ndarray:
        h, w = arr.shape[:2]
        if h < size or w < size:
            raise ValueError(f"Image/mask smaller than crop size {size}: got {(h, w)}")

        top = (h - size) // 2
        left = (w - size) // 2
        return arr[top:top + size, left:left + size]

    @staticmethod
    def map_s2ds_label(gt: np.ndarray) -> np.ndarray:
        """
        S2DS label rule:
        crack: 255 -> 1
        ignore/other defects: 1~254 -> 255
        background: 0 -> 0
        """
        target = np.zeros_like(gt, dtype=np.uint8)
        target[gt == 255] = 1
        target[(gt > 0) & (gt < 255)] = 255
        return target

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        img_path, gt_path, name = self.samples[index]

        image_rgb = np.asarray(Image.open(img_path).convert("RGB"))
        gt = np.asarray(Image.open(gt_path).convert("L"))

        image_rgb = self.center_crop_np(image_rgb, self.train_size)
        gt = self.center_crop_np(gt, self.train_size)
        target = self.map_s2ds_label(gt)

        image_tensor = self.img_transform(Image.fromarray(image_rgb))

        return {
            "image": image_tensor,
            "image_rgb": torch.from_numpy(image_rgb.copy()).permute(2, 0, 1),
            "label": torch.from_numpy(target).unsqueeze(0),
            "name": name,
        }


@contextlib.contextmanager
def import_repo(repo_root: str):
    if not os.path.isdir(repo_root):
        raise FileNotFoundError(f"Repository root not found: {repo_root}")

    old_sys_path = list(sys.path)

    managed_prefixes = ("model", "models", "datasets")
    saved_modules = {
        name: module
        for name, module in sys.modules.items()
        if any(name == p or name.startswith(p + ".") for p in managed_prefixes)
    }

    for name in list(saved_modules.keys()):
        sys.modules.pop(name, None)

    sys.path.insert(0, repo_root)

    try:
        yield
    finally:
        sys.path[:] = old_sys_path

        for name in list(sys.modules.keys()):
            if any(name == p or name.startswith(p + ".") for p in managed_prefixes):
                sys.modules.pop(name, None)

        sys.modules.update(saved_modules)


def load_checkpoint_file(path: str) -> Any:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    try:
        return torch.load(path, map_location="cpu")
    except TypeError:
        return torch.load(path, map_location="cpu", weights_only=False)
    except Exception:
        return torch.load(path, map_location="cpu", weights_only=False)


def extract_state_dict(ckpt: Any) -> Dict[str, torch.Tensor]:
    if isinstance(ckpt, torch.nn.Module):
        return ckpt.state_dict()

    if not isinstance(ckpt, dict):
        raise TypeError(f"Unsupported checkpoint type: {type(ckpt)}")

    for key in ["state_dict", "model_state_dict", "net", "model", "weights"]:
        if key in ckpt and isinstance(ckpt[key], dict):
            ckpt = ckpt[key]
            break

    clean = {}
    for k, v in ckpt.items():
        if not isinstance(v, torch.Tensor):
            continue
        nk = k[7:] if k.startswith("module.") else k
        clean[nk] = v

    return clean


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


def build_official_scsegamba(repo_root: str, device: str):
    with import_repo(repo_root):
        from models import build

        sc_args = types.SimpleNamespace(
            device=device,
            Norm_Type="GN",
            BCELoss_ratio=0.83,
            DiceLoss_ratio=0.17,
            phase="val",
        )

        model, _ = build(sc_args)
        return model


def load_weights(model: torch.nn.Module, ckpt_path: str):
    ckpt = load_checkpoint_file(ckpt_path)
    state_dict = extract_state_dict(ckpt)

    msg = model.load_state_dict(state_dict, strict=False)

    missing = list(getattr(msg, "missing_keys", []))
    unexpected = list(getattr(msg, "unexpected_keys", []))

    if missing:
        print(f"  Missing keys: {len(missing)}")
    if unexpected:
        print(f"  Unexpected keys: {len(unexpected)}")


def colorize_gt(gt: np.ndarray) -> np.ndarray:
    """
    black: background
    white: crack
    gray: ignored regions
    """
    out = np.zeros((gt.shape[0], gt.shape[1], 3), dtype=np.uint8)
    out[gt == 1] = [255, 255, 255]
    out[gt == 255] = [120, 120, 120]
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
    for p in panels:
        canvas.paste(p, (x, 0))
        x += p.width

    canvas.save(save_path)


def ensure_dirs(save_root: str, fold: int):
    dirs = {
        "pred": os.path.join(save_root, "pred_png", f"fold{fold}"),
        "prob": os.path.join(save_root, "prob_png", f"fold{fold}"),
        "overlay": os.path.join(save_root, "overlay_png", f"fold{fold}"),
        "compare": os.path.join(save_root, "compare_png", f"fold{fold}"),
    }

    for d in dirs.values():
        os.makedirs(d, exist_ok=True)

    return dirs


def generate_one_fold(
    fold: int,
    ckpt_path: str,
    args,
):
    print(f"\n{'=' * 88}")
    print(f"Fold {fold}")
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
        pin_memory=True,
    )

    model = build_official_scsegamba(args.scsegamba_root, args.device)
    load_weights(model, ckpt_path)

    model = model.to(args.device)
    model.eval()

    dirs = ensure_dirs(args.save_root, fold)

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

                valid = gt != 255
                pred_i[~valid] = 0

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

    print(f"Saved {count} samples to: {os.path.abspath(args.save_root)}")

    del model
    if torch.cuda.is_available() and str(args.device).startswith("cuda"):
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--s2ds-root",
        default="/home/skye/data/Skye/databases/s2ds5",
    )
    parser.add_argument(
        "--scsegamba-root",
        default="/home/skye/data/Skye/third_party/SCSegamba",
    )
    parser.add_argument(
        "--save-root",
        default="/home/skye/data/Skye/third_party/SCSegamba/results/SCSegamba_Baseline_Sequential/synth2real/predictions_s2ds5",
    )

    parser.add_argument("--split", default="val", choices=["train", "val", "test"])
    parser.add_argument("--folds", nargs="*", type=int, default=[1, 2, 3, 4, 5])
    parser.add_argument("--train-size", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--overlay-alpha", type=float, default=0.55)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")

    return parser.parse_args()


def main():
    args = parse_args()

    for fold in args.folds:
        if fold not in SCSegamba_SYNTH2REAL_CKPTS:
            raise ValueError(f"Invalid fold: {fold}")

        generate_one_fold(
            fold=fold,
            ckpt_path=SCSegamba_SYNTH2REAL_CKPTS[fold],
            args=args,
        )

    print("\nDone.")
    print(f"Prediction root: {os.path.abspath(args.save_root)}")


if __name__ == "__main__":
    main()