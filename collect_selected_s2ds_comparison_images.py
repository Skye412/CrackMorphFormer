#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Collect selected S2DS samples into per-sample folders.

Each sample folder contains exactly nine 512x512 PNGs:
  01_image.png
  02_gt.png
  03_unet.png
  04_fcn.png
  05_deeplabv3plus.png
  06_segformer_b2.png
  07_wpformer.png
  08_scsegamba.png
  09_crackmorphformer.png
"""

from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
from PIL import Image


DATA_ROOT = Path("/home/skye/data/Skye/databases/s2ds5")
OUT_ROOT = Path(
    "/home/skye/data/Skye/CrackMorphFormer/results/"
    "CrackMorphFormer_final_stable/predictions_s2ds5/selected_comparison_samples"
)

METHOD_ROOTS: List[Tuple[str, str, Path]] = [
    (
        "03_unet.png",
        "U-Net",
        Path("/home/skye/data/Skye/CrackMorphFormer/alltheline/"
             "s2ds_prediction_vis_all_5fold/unet_synth2real"),
    ),
    (
        "04_fcn.png",
        "FCN",
        Path("/home/skye/data/Skye/CrackMorphFormer/alltheline/"
             "s2ds_prediction_vis_all_5fold/fcn_resnet50_synth2real"),
    ),
    (
        "05_deeplabv3plus.png",
        "DeepLabV3+",
        Path("/home/skye/data/Skye/CrackMorphFormer/alltheline/"
             "s2ds_prediction_vis_all_5fold/deeplabv3plus_synth2real"),
    ),
    (
        "06_segformer_b2.png",
        "SegFormer-B2",
        Path("/home/skye/data/Skye/CrackMorphFormer/alltheline/"
             "s2ds_prediction_vis_all_5fold/segformer_b2_synth2real"),
    ),
    (
        "07_wpformer.png",
        "WPFormer",
        Path("/home/skye/data/Skye/CrackMorphFormer/alltheline/"
             "s2ds_prediction_vis_all_5fold/wpformer_synth2real"),
    ),
    (
        "08_scsegamba.png",
        "SCSegamba",
        Path("/home/skye/data/Skye/third_party/SCSegamba/results/"
             "SCSegamba_Baseline_Sequential/synth2real/predictions_s2ds5"),
    ),
    (
        "09_crackmorphformer.png",
        "CrackMorphFormer",
        Path("/home/skye/data/Skye/CrackMorphFormer/results/"
             "CrackMorphFormer_final_stable/predictions_s2ds5/synth2real_full"),
    ),
]


def read_val_folds(data_root: Path) -> Dict[str, int]:
    folds: Dict[str, int] = {}
    for fold in range(1, 6):
        split_path = data_root / f"fold{fold}_val.txt"
        if not split_path.is_file():
            raise FileNotFoundError(f"Split file not found: {split_path}")
        for line in split_path.read_text(encoding="utf-8").splitlines():
            name = line.strip()
            if name:
                folds[Path(name).stem] = fold
    return folds


def build_prediction_index(root: Path) -> Dict[Tuple[int, str], Path]:
    if not root.is_dir():
        raise FileNotFoundError(f"Method directory not found: {root}")

    index: Dict[Tuple[int, str], Path] = {}

    pred_png_root = root / "pred_png"
    if pred_png_root.is_dir():
        for path in pred_png_root.glob("fold*/*.png"):
            try:
                fold = int(path.parent.name.replace("fold", ""))
            except ValueError:
                continue
            index[(fold, path.stem)] = path

    for path in root.glob("fold*/*/pred.png"):
        try:
            fold = int(path.parent.parent.name.replace("fold", ""))
        except ValueError:
            continue
        index[(fold, path.parent.name)] = path

    for sample_dir in root.glob("fold*/*"):
        if not sample_dir.is_dir():
            continue
        try:
            fold = int(sample_dir.parent.name.replace("fold", ""))
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

    return index


def center_crop(arr: np.ndarray, size: int) -> np.ndarray:
    h, w = arr.shape[:2]
    if h < size or w < size:
        raise ValueError(f"Array smaller than crop size {size}: got {(h, w)}")
    top = (h - size) // 2
    left = (w - size) // 2
    return arr[top:top + size, left:left + size]


def save_image(data_root: Path, stem: str, out_path: Path, size: int):
    src = data_root / "images" / f"{stem}.png"
    if not src.is_file():
        raise FileNotFoundError(f"Image not found: {src}")
    arr = np.asarray(Image.open(src).convert("RGB"))
    Image.fromarray(center_crop(arr, size)).save(out_path)


def save_gt(data_root: Path, stem: str, out_path: Path, size: int):
    src = data_root / "labs" / f"{stem}.png"
    if not src.is_file():
        raise FileNotFoundError(f"GT not found: {src}")
    gt_rgb = np.asarray(Image.open(src).convert("RGB"))
    gt_rgb = center_crop(gt_rgb, size)
    crack = np.all(gt_rgb == np.array([255, 255, 255], dtype=np.uint8), axis=-1)
    out = np.zeros((*crack.shape, 3), dtype=np.uint8)
    out[crack] = [255, 255, 255]
    Image.fromarray(out).save(out_path)


def save_prediction(src: Path, out_path: Path, size: int):
    arr = np.asarray(Image.open(src).convert("L"))
    if arr.shape != (size, size):
        arr = np.asarray(
            Image.fromarray(arr).resize((size, size), Image.Resampling.NEAREST)
        )
    pred = (arr > 127).astype(np.uint8) * 255
    Image.fromarray(pred).save(out_path)


def parse_ids(values: Iterable[str]) -> List[str]:
    stems = []
    for value in values:
        if value.startswith("s2ds_"):
            stems.append(value)
        else:
            stems.append(f"s2ds_{int(value):03d}")
    return stems


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ids",
        nargs="+",
        default=["2", "24", "75", "152", "216", "276", "312", "288", "684"],
    )
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument("--out-root", type=Path, default=OUT_ROOT)
    parser.add_argument("--size", type=int, default=512)
    args = parser.parse_args()

    stems = parse_ids(args.ids)
    val_folds = read_val_folds(args.data_root)
    method_indices = [
        (filename, label, build_prediction_index(root))
        for filename, label, root in METHOD_ROOTS
    ]

    args.out_root.mkdir(parents=True, exist_ok=True)

    report_rows = []
    for stem in stems:
        if stem not in val_folds:
            raise ValueError(f"{stem} not found in S2DS val folds")

        fold = val_folds[stem]
        sample_dir = args.out_root / stem
        if sample_dir.exists():
            shutil.rmtree(sample_dir)
        sample_dir.mkdir(parents=True, exist_ok=True)

        save_image(args.data_root, stem, sample_dir / "01_image.png", args.size)
        save_gt(args.data_root, stem, sample_dir / "02_gt.png", args.size)

        missing = []
        for filename, label, index in method_indices:
            src: Optional[Path] = index.get((fold, stem))
            if src is None:
                missing.append(label)
                continue
            save_prediction(src, sample_dir / filename, args.size)

        report_rows.append({
            "sample": stem,
            "fold": fold,
            "file_count": len(list(sample_dir.glob("*.png"))),
            "missing_methods": ";".join(missing),
            "folder": str(sample_dir),
        })

    report_path = args.out_root / "selected_samples_report.csv"
    with report_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["sample", "fold", "file_count", "missing_methods", "folder"],
        )
        writer.writeheader()
        writer.writerows(report_rows)

    print(f"Saved {len(stems)} sample folders to: {args.out_root}")
    print(f"Report: {report_path}")


if __name__ == "__main__":
    main()
