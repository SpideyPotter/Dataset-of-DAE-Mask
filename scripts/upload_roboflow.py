#!/usr/bin/env python3
"""Upload YOLO dataset to Roboflow."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def load_api_key(explicit_key: str | None = None) -> str | None:
    if explicit_key:
        return explicit_key

    env_path = Path("/workspace/.env")
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            if key.strip() == "ROBOFLOW_API_KEY" and value.strip():
                return value.strip().strip('"').strip("'")

    for name in ("ROBOFLOW_API_KEY", "ROBOFLOW_KEY"):
        value = os.environ.get(name)
        if value:
            return value

    try:
        from roboflow.config import load_roboflow_api_key

        return load_roboflow_api_key()
    except Exception:
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Upload YOLO dataset to Roboflow")
    parser.add_argument("--dataset", type=Path, default=Path("/workspace/yolo_dataset"))
    parser.add_argument("--project", default="mswdd2022-wheat")
    parser.add_argument("--workspace", default=None, help="Roboflow workspace slug (optional)")
    parser.add_argument("--api-key", default=None, help="Roboflow API key override")
    parser.add_argument("--batch-name", default="mswdd2022-yolo-import")
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--zip", action="store_true", help="Use zip upload flow")
    args = parser.parse_args()

    api_key = load_api_key(args.api_key)
    if not api_key:
        print("Error: Roboflow API key not found.", file=sys.stderr)
        print("Set ROBOFLOW_API_KEY, add it to /workspace/.env, or pass --api-key.", file=sys.stderr)
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

    workspace_slug = workspace.url.rstrip("/").split("/")[-1]
    project_slug = args.project

    print("Upload finished.")
    print(f"Project: {project_slug}")
    print(f"Workspace: {workspace.name} ({workspace_slug})")
    print(f"URL: https://app.roboflow.com/{workspace_slug}/{project_slug}")

    try:
        project = workspace.project(project_slug)
        info = project.info if hasattr(project, "info") else {}
        if isinstance(info, dict):
            print(f"Images in project: {info.get('images', 'unknown')}")
            print(f"Classes: {list((info.get('classes') or {}).keys())}")
    except Exception as exc:
        print(f"Note: could not fetch project details ({exc})")

    if result:
        print(f"Result: {result}")


if __name__ == "__main__":
    main()
