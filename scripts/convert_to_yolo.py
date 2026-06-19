#!/usr/bin/env python3
"""Convert MSWDD2022 LabelMe polygon annotations to YOLO detection format."""

from __future__ import annotations

import argparse
import json
import random
import shutil
from pathlib import Path

CLASS_MAP = {
    "Wheat yellow drawf": 0,
    "wheat powdery mildew": 1,
    "wheat scab": 2,
    "wheat stripe rust": 3,
}

CLASS_NAMES = [
    "yellow_dwarf",
    "powdery_mildew",
    "scab",
    "stripe_rust",
]

PREFIX_MAP = {
    "Wheat yellow drawf": "yellow_dwarf",
    "wheat powdery mildew": "powdery_mildew",
    "wheat scab": "scab",
    "wheat stripe rust": "stripe_rust",
}


def polygon_to_yolo_bbox(points: list[list[float]], img_w: int, img_h: int) -> tuple[float, float, float, float]:
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)

    x_center = ((x_min + x_max) / 2) / img_w
    y_center = ((y_min + y_max) / 2) / img_h
    width = (x_max - x_min) / img_w
    height = (y_max - y_min) / img_h

    return x_center, y_center, width, height


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def convert_sample(
    json_path: Path,
    img_path: Path,
    class_id: int,
) -> list[str]:
    with json_path.open() as f:
        data = json.load(f)

    img_w = data.get("imageWidth")
    img_h = data.get("imageHeight")
    if not img_w or not img_h:
        raise ValueError(f"Missing image dimensions in {json_path}")

    lines: list[str] = []
    for shape in data.get("shapes", []):
        if shape.get("shape_type") != "polygon":
            continue
        points = shape.get("points", [])
        if len(points) < 3:
            continue

        x_c, y_c, w, h = polygon_to_yolo_bbox(points, img_w, img_h)
        lines.append(
            f"{class_id} {clamp01(x_c):.6f} {clamp01(y_c):.6f} {clamp01(w):.6f} {clamp01(h):.6f}"
        )

    if not lines:
        raise ValueError(f"No valid polygon annotations in {json_path}")

    if not img_path.exists():
        raise FileNotFoundError(f"Missing image for {json_path}: {img_path}")

    return lines


def split_items(items: list[tuple[str, Path, Path]], seed: int, train: float, val: float) -> dict[str, list]:
    by_class: dict[str, list] = {name: [] for name in CLASS_MAP}
    for category, json_path, img_path in items:
        by_class[category].append((category, json_path, img_path))

    splits = {"train": [], "valid": [], "test": []}
    rng = random.Random(seed)

    for category, samples in by_class.items():
        rng.shuffle(samples)
        n = len(samples)
        n_train = int(n * train)
        n_val = int(n * val)
        n_test = n - n_train - n_val

        splits["train"].extend(samples[:n_train])
        splits["valid"].extend(samples[n_train : n_train + n_val])
        splits["test"].extend(samples[n_train + n_val :])

    for split_name in splits:
        rng.shuffle(splits[split_name])

    return splits


def write_data_yaml(output_dir: Path) -> None:
    content = "\n".join(
        [
            f"path: {output_dir.resolve()}",
            "train: train/images",
            "val: valid/images",
            "test: test/images",
            f"nc: {len(CLASS_NAMES)}",
            f"names: {CLASS_NAMES}",
            "",
        ]
    )
    (output_dir / "data.yaml").write_text(content)


def convert_dataset(
    source_dir: Path,
    output_dir: Path,
    seed: int = 42,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
) -> dict[str, int]:
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)

    items: list[tuple[str, Path, Path]] = []
    for category in CLASS_MAP:
        img_dir = source_dir / category / "img"
        json_dir = source_dir / category / "json"
        for json_path in sorted(json_dir.glob("*.json")):
            stem = json_path.stem
            img_path = img_dir / f"{stem}.png"
            items.append((category, json_path, img_path))

    splits = split_items(items, seed=seed, train=train_ratio, val=val_ratio)
    counts = {"train": 0, "valid": 0, "test": 0, "annotations": 0}

    for split_name, samples in splits.items():
        images_dir = output_dir / split_name / "images"
        labels_dir = output_dir / split_name / "labels"
        images_dir.mkdir(parents=True, exist_ok=True)
        labels_dir.mkdir(parents=True, exist_ok=True)

        for category, json_path, img_path in samples:
            class_id = CLASS_MAP[category]
            prefix = PREFIX_MAP[category]
            stem = json_path.stem
            out_name = f"{prefix}_{stem}"

            lines = convert_sample(json_path, img_path, class_id)
            shutil.copy2(img_path, images_dir / f"{out_name}.png")
            (labels_dir / f"{out_name}.txt").write_text("\n".join(lines) + "\n")

            counts[split_name] += 1
            counts["annotations"] += len(lines)

    write_data_yaml(output_dir)
    counts["total"] = sum(counts[k] for k in ("train", "valid", "test"))
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert MSWDD2022 LabelMe dataset to YOLO format")
    parser.add_argument("--source", type=Path, default=Path("/workspace"))
    parser.add_argument("--output", type=Path, default=Path("/workspace/yolo_dataset"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    counts = convert_dataset(args.source, args.output, seed=args.seed)
    print("Conversion complete:")
    for key, value in counts.items():
        print(f"  {key}: {value}")
    print(f"Output: {args.output}")


if __name__ == "__main__":
    main()
