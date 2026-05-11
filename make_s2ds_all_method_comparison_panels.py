#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Build side-by-side S2DS comparison panels across all synth2real methods.

Panel order:
  Image | GT | U-Net | FCN | DeepLabV3+ | SegFormer-B2 | WPFormer | SCSegamba | CrackMorphFormer

The script handles both prediction layouts used in this workspace:
  1. <method>/fold*/<sample_stem>/pred.png
  2. <method>/pred_png/fold*/<sample_stem>.png
"""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw


DATA_ROOT = Path("/home/skye/data/Skye/databases/s2ds5")
OUT_ROOT = Path(
    "/home/skye/data/Skye/CrackMorphFormer/results/"
    "CrackMorphFormer_final_stable/predictions_s2ds5/all_method_comparison_panels"
)


METHOD_ROOTS: List[Tuple[str, Path]] = [
    (
        "U-Net",
        Path("/home/skye/data/Skye/CrackMorphFormer/alltheline/"
             "s2ds_prediction_vis_all_5fold/unet_synth2real"),
    ),
    (
        "FCN",
        Path("/home/skye/data/Skye/CrackMorphFormer/alltheline/"
             "s2ds_prediction_vis_all_5fold/fcn_resnet50_synth2real"),
    ),
    (
        "DeepLabV3+",
        Path("/home/skye/data/Skye/CrackMorphFormer/alltheline/"
             "s2ds_prediction_vis_all_5fold/deeplabv3plus_synth2real"),
    ),
    (
        "SegFormer-B2",
        Path("/home/skye/data/Skye/CrackMorphFormer/alltheline/"
             "s2ds_prediction_vis_all_5fold/segformer_b2_synth2real"),
    ),
    (
        "WPFormer",
        Path("/home/skye/data/Skye/CrackMorphFormer/alltheline/"
             "s2ds_prediction_vis_all_5fold/wpformer_synth2real"),
    ),
    (
        "SCSegamba",
        Path("/home/skye/data/Skye/third_party/SCSegamba/results/"
             "SCSegamba_Baseline_Sequential/synth2real/predictions_s2ds5"),
    ),
    (
        "CrackMorphFormer",
        Path("/home/skye/data/Skye/CrackMorphFormer/results/"
             "CrackMorphFormer_final_stable/predictions_s2ds5/synth2real_full"),
    ),
]


def read_split_samples(data_root: Path, folds: Iterable[int], split: str) -> List[Tuple[int, str]]:
    samples: List[Tuple[int, str]] = []
    for fold in folds:
        split_path = data_root / f"fold{fold}_{split}.txt"
        if not split_path.is_file():
            raise FileNotFoundError(f"Split file not found: {split_path}")

        with split_path.open("r", encoding="utf-8") as f:
            for line in f:
                name = line.strip()
                if name:
                    samples.append((fold, Path(name).stem))

    return samples


def build_prediction_index(root: Path) -> Dict[Tuple[int, str], Path]:
    if not root.is_dir():
        raise FileNotFoundError(f"Method directory not found: {root}")

    index: Dict[Tuple[int, str], Path] = {}

    pred_png_root = root / "pred_png"
    if pred_png_root.is_dir():
        for path in pred_png_root.glob("fold*/*.png"):
            fold_name = path.parent.name
            if not fold_name.startswith("fold"):
                continue
            try:
                fold = int(fold_name.replace("fold", ""))
            except ValueError:
                continue
            index[(fold, path.stem)] = path

    for path in root.glob("fold*/*/pred.png"):
        fold_name = path.parent.parent.name
        if not fold_name.startswith("fold"):
            continue
        try:
            fold = int(fold_name.replace("fold", ""))
        except ValueError:
            continue
        index[(fold, path.parent.name)] = path

    for sample_dir in root.glob("fold*/*"):
        if not sample_dir.is_dir():
            continue

        fold_name = sample_dir.parent.name
        if not fold_name.startswith("fold"):
            continue

        try:
            fold = int(fold_name.replace("fold", ""))
        except ValueError:
            continue

        key = (fold, sample_dir.name)
        if key in index:
            continue

        candidates = [
            p for p in sample_dir.glob("*.png")
            if p.name not in {
                "input.png",
                "gt.png",
                "gt_overlay.png",
                "pred_overlay.png",
                "panel.png",
            }
        ]
        if candidates:
            index[key] = sorted(candidates)[0]

    # Fallback for slightly different layouts: prefer explicit pred.png files,
    # then direct fold*/sample.png files. Existing keys are kept.
    for path in root.rglob("pred.png"):
        parts = path.parts
        fold = None
        for part in parts:
            if part.startswith("fold"):
                try:
                    fold = int(part.replace("fold", ""))
                    break
                except ValueError:
                    pass
        if fold is not None:
            index.setdefault((fold, path.parent.name), path)

    for path in root.glob("fold*/*.png"):
        fold_name = path.parent.name
        if not fold_name.startswith("fold"):
            continue
        try:
            fold = int(fold_name.replace("fold", ""))
        except ValueError:
            continue
        index.setdefault((fold, path.stem), path)

    return index


def center_crop(arr: np.ndarray, size: int) -> np.ndarray:
    h, w = arr.shape[:2]
    if h < size or w < size:
        raise ValueError(f"Array smaller than crop size {size}: got {(h, w)}")
    top = (h - size) // 2
    left = (w - size) // 2
    return arr[top:top + size, left:left + size]


def load_image_panel(data_root: Path, stem: str, size: int) -> Image.Image:
    path = data_root / "images" / f"{stem}.png"
    if not path.is_file():
        raise FileNotFoundError(f"Image not found: {path}")
    arr = np.asarray(Image.open(path).convert("RGB"))
    arr = center_crop(arr, size)
    return Image.fromarray(arr)


def load_gt_panel(data_root: Path, stem: str, size: int) -> Image.Image:
    path = data_root / "labs" / f"{stem}.png"
    if not path.is_file():
        raise FileNotFoundError(f"GT not found: {path}")
    gt_rgb = np.asarray(Image.open(path).convert("RGB"))
    gt_rgb = center_crop(gt_rgb, size)
    crack = np.all(gt_rgb == np.array([255, 255, 255], dtype=np.uint8), axis=-1)
    out = np.zeros((*crack.shape, 3), dtype=np.uint8)
    out[crack] = [255, 255, 255]
    return Image.fromarray(out)


def normalize_prediction(path: Path, size: int) -> Image.Image:
    arr = np.asarray(Image.open(path).convert("L"))

    if arr.shape[0] != size or arr.shape[1] != size:
        img = Image.fromarray(arr)
        img = img.resize((size, size), Image.Resampling.NEAREST)
        arr = np.asarray(img)

    pred = (arr > 127).astype(np.uint8) * 255
    rgb = np.repeat(pred[..., None], 3, axis=2)
    return Image.fromarray(rgb)


def missing_panel(size: int) -> Image.Image:
    img = Image.new("RGB", (size, size), (245, 245, 245))
    draw = ImageDraw.Draw(img)
    draw.line((0, 0, size, size), fill=(180, 180, 180), width=4)
    draw.line((0, size, size, 0), fill=(180, 180, 180), width=4)
    draw.text((16, 16), "missing", fill=(80, 80, 80))
    return img


def add_title(img: Image.Image, title: str, title_h: int) -> Image.Image:
    canvas = Image.new("RGB", (img.width, img.height + title_h), "white")
    canvas.paste(img, (0, title_h))
    draw = ImageDraw.Draw(canvas)
    draw.text((8, 8), title, fill=(0, 0, 0))
    return canvas


def make_panel(columns: List[Tuple[str, Image.Image]], title_h: int) -> Image.Image:
    titled = [add_title(img, title, title_h) for title, img in columns]
    width = sum(img.width for img in titled)
    height = max(img.height for img in titled)
    canvas = Image.new("RGB", (width, height), "white")

    x = 0
    for img in titled:
        canvas.paste(img, (x, 0))
        x += img.width

    return canvas


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument("--out-root", type=Path, default=OUT_ROOT)
    parser.add_argument("--folds", nargs="+", type=int, default=[1, 2, 3, 4, 5])
    parser.add_argument("--split", default="val", choices=["train", "val"])
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--title-height", type=int, default=32)
    args = parser.parse_args()

    samples = read_split_samples(args.data_root, args.folds, args.split)
    method_indices = [(name, build_prediction_index(root)) for name, root in METHOD_ROOTS]

    for fold in args.folds:
        (args.out_root / f"fold{fold}").mkdir(parents=True, exist_ok=True)
    (args.out_root / "all").mkdir(parents=True, exist_ok=True)

    rows = []
    for fold, stem in samples:
        columns: List[Tuple[str, Image.Image]] = [
            ("Image", load_image_panel(args.data_root, stem, args.size)),
            ("GT", load_gt_panel(args.data_root, stem, args.size)),
        ]

        missing_methods = []
        for method_name, index in method_indices:
            pred_path: Optional[Path] = index.get((fold, stem))
            if pred_path is None:
                columns.append((method_name, missing_panel(args.size)))
                missing_methods.append(method_name)
            else:
                columns.append((method_name, normalize_prediction(pred_path, args.size)))

        panel = make_panel(columns, args.title_height)

        fold_path = args.out_root / f"fold{fold}" / f"{stem}.png"
        all_path = args.out_root / "all" / f"fold{fold}_{stem}.png"
        panel.save(fold_path)
        panel.save(all_path)

        rows.append({
            "fold": fold,
            "sample": stem,
            "missing_methods": ";".join(missing_methods),
            "panel": str(fold_path),
        })

    report_path = args.out_root / "missing_report.csv"
    with report_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["fold", "sample", "missing_methods", "panel"])
        writer.writeheader()
        writer.writerows(rows)

    missing_count = sum(1 for row in rows if row["missing_methods"])
    print(f"Saved {len(rows)} comparison panels to: {args.out_root}")
    print(f"Flat copy: {args.out_root / 'all'}")
    print(f"Missing report: {report_path}")
    print(f"Samples with missing methods: {missing_count}")


if __name__ == "__main__":
    main()
