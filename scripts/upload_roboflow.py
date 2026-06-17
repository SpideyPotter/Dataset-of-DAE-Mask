#!/usr/bin/env python3
"""Upload YOLO dataset to Roboflow."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Upload YOLO dataset to Roboflow")
    parser.add_argument("--dataset", type=Path, default=Path("/workspace/yolo_dataset"))
    parser.add_argument("--project", default="mswdd2022-wheat-diseases")
    parser.add_argument("--workspace", default=None, help="Roboflow workspace slug (optional)")
    parser.add_argument("--batch-name", default="mswdd2022-yolo-import")
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--zip", action="store_true", help="Use zip upload flow")
    args = parser.parse_args()

    api_key = os.environ.get("ROBOFLOW_API_KEY")
    if not api_key:
        print("Error: ROBOFLOW_API_KEY environment variable is not set.", file=sys.stderr)
        print("Get your key at https://app.roboflow.com/settings/api", file=sys.stderr)
        sys.exit(1)

    if not args.dataset.exists():
        print(f"Error: dataset path does not exist: {args.dataset}", file=sys.stderr)
        sys.exit(1)

    import roboflow

    rf = roboflow.Roboflow(api_key=api_key)
    workspace = rf.workspace(args.workspace) if args.workspace else rf.workspace()

    print(f"Uploading {args.dataset} to project '{args.project}'...")
    result = workspace.upload_dataset(
        str(args.dataset),
        args.project,
        num_workers=args.workers,
        project_license="MIT",
        project_type="object-detection",
        batch_name=args.batch_name,
        use_zip_upload=args.zip,
    )

    project = workspace.project(args.project)
    workspace_slug = workspace.url.rstrip("/").split("/")[-1]
    print("Upload finished.")
    print(f"Project: {args.project}")
    print(f"Workspace: {workspace.name} ({workspace_slug})")
    print(f"URL: https://app.roboflow.com/{workspace_slug}/{args.project}")
    if result:
        print(f"Result: {result}")


if __name__ == "__main__":
    main()
