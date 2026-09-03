#!/usr/bin/env python3
"""Create compact-view experiments by uniformly sampling the first 60 source frames."""

import json
import shutil
from pathlib import Path

import numpy as np

SOURCE = Path("/root/multiview_compare/experiments/random")
TARGET = Path("/root/multiview_compare/experiments/short60")


for scene in sorted(SOURCE.glob("[0-9][0-9]_*")):
    if not scene.is_dir():
        continue
    seed = next((p for p in (scene / "16views" / "querysplat" / "input").glob("*.png")), None)
    if seed is None:
        raise RuntimeError(f"Cannot locate source images for {scene.name}")
    image_dir = seed.resolve().parent
    images = sorted(image_dir.glob("*.png"))
    if len(images) < 60:
        raise RuntimeError(f"{scene.name} has only {len(images)} source images")
    pool = images[:60]
    scene_target = TARGET / scene.name
    scene_target.mkdir(parents=True, exist_ok=True)
    transforms = scene / "transforms.json"
    if transforms.is_file() and not (scene_target / "transforms.json").exists():
        shutil.copy2(transforms, scene_target / "transforms.json")
    for views in (4, 8, 12, 16):
        indices = np.rint(np.linspace(0, 59, views)).astype(int).tolist()
        chosen = [pool[i] for i in indices]
        experiment = scene_target / f"{views}views" / "querysplat"
        input_dir = experiment / "input"
        input_dir.mkdir(parents=True, exist_ok=True)
        for image in chosen:
            link = input_dir / image.name
            if not link.exists():
                link.symlink_to(image.resolve())
        manifest = {
            "policy": "uniform sampling from first 60 frames; no timestamps available",
            "source_image_directory": str(image_dir),
            "pool_first": pool[0].name,
            "pool_last": pool[-1].name,
            "pool_size": 60,
            "views": views,
            "zero_based_pool_indices": indices,
            "selected_frames": [p.name for p in chosen],
        }
        (experiment / "selection_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        print(scene.name, views, ",".join(p.stem for p in chosen))
