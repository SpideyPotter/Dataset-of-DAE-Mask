#!/usr/bin/env python3
"""Upload only images missing from the Roboflow project."""

from __future__ import annotations

import argparse
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from upload_roboflow import load_api_key

CLASS_NAMES = [
    "yellow_dwarf",
    "powdery_mildew",
    "scab",
    "stripe_rust",
]

LABELMAP_PATH = Path("/workspace/yolo_dataset/labelmap.txt")


def write_labelmap() -> Path:
    LABELMAP_PATH.write_text("\n".join(CLASS_NAMES) + "\n")
    return LABELMAP_PATH


def list_remote_names(project) -> set[str]:
    names: set[str] = set()
    offset = 0
    limit = 250
    while True:
        page = project.search(offset=offset, limit=limit, fields=["name"])
        if not page:
            break
        for img in page:
            if img.get("name"):
                names.add(img["name"])
        if len(page) < limit:
            break
        offset += limit
    return names


def find_missing(dataset_dir: Path, remote_names: set[str]) -> list[tuple[str, Path, Path]]:
    missing: list[tuple[str, Path, Path]] = []
    for split in ("train", "valid", "test"):
        images_dir = dataset_dir / split / "images"
        labels_dir = dataset_dir / split / "labels"
        for image_path in sorted(images_dir.glob("*.png")):
            if image_path.name not in remote_names:
                label_path = labels_dir / f"{image_path.stem}.txt"
                missing.append((split, image_path, label_path))
    return missing


def upload_one(project, split: str, image_path: Path, label_path: Path, batch_name: str) -> tuple[str, bool, str]:
    try:
        project.single_upload(
            image_path=str(image_path),
            annotation_path=str(label_path),
            annotation_labelmap=str(LABELMAP_PATH),
            split=split,
            batch_name=batch_name,
            num_retry_uploads=2,
        )
        return image_path.name, True, "ok"
    except Exception as exc:
        return image_path.name, False, str(exc)


def main() -> None:
    parser = argparse.ArgumentParser(description="Upload missing YOLO images to Roboflow")
    parser.add_argument("--dataset", type=Path, default=Path("/workspace/yolo_dataset"))
    parser.add_argument("--project", default="mswdd2022-wheat")
    parser.add_argument("--batch-name", default="mswdd2022-missing-upload")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    api_key = load_api_key()
    if not api_key:
        print("Error: Roboflow API key not found.", file=sys.stderr)
        sys.exit(1)

    import roboflow

    write_labelmap()
    rf = roboflow.Roboflow(api_key=api_key)
    project = rf.workspace().project(args.project)

    print("Fetching remote image list...")
    remote = list_remote_names(project)
    print(f"Remote images: {len(remote)}")

    missing = find_missing(args.dataset, remote)
    print(f"Missing images to upload: {len(missing)}")
    if not missing:
        print("Nothing to upload.")
        return

    ok = 0
    failed: list[tuple[str, str]] = []

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [
            pool.submit(upload_one, project, split, image_path, label_path, args.batch_name)
            for split, image_path, label_path in missing
        ]
        for i, future in enumerate(as_completed(futures), 1):
            name, success, msg = future.result()
            if success:
                ok += 1
                print(f"[{i}/{len(missing)}] UPLOADED {name}")
            else:
                failed.append((name, msg))
                print(f"[{i}/{len(missing)}] FAILED {name}: {msg}")
            if i % 50 == 0:
                time.sleep(0.5)

    remote_after = list_remote_names(project)
    print(f"\nDone. Uploaded: {ok}, Failed: {len(failed)}, Remote now: {len(remote_after)}")
    if failed:
        fail_path = Path("/workspace/upload_failed.txt")
        fail_path.write_text("\n".join(f"{n}\t{e}" for n, e in failed))
        print(f"Failures written to {fail_path}")


if __name__ == "__main__":
    main()
