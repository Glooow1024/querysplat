#!/usr/bin/env python3
"""Filter existing samples to frames with GT poses and write normalized camera priors."""

import json
import os
import shutil
from pathlib import Path

import numpy as np
from PIL import Image


SOURCE = Path(os.environ.get("DL3DV_EXPERIMENT_SOURCE", "/root/multiview_compare/experiments/random"))
TARGET = Path(os.environ.get("DL3DV_GT_SUBSET_TARGET", "/root/multiview_compare/inputs/random_gt"))
SIZE = 256
CV_FLIP = np.diag([1.0, -1.0, -1.0, 1.0])


for scene in sorted(SOURCE.glob("[0-9][0-9]_*")):
    if not scene.is_dir():
        continue
    tf = json.loads((scene / "transforms.json").read_text())
    gt = {Path(frame["file_path"]).stem: frame for frame in tf.get("frames", [])}
    for nominal in (4, 8, 12, 14, 16):
        source_images = sorted((scene / f"{nominal}views" / "querysplat" / "input").glob("*.png"))
        matched = [path for path in source_images if path.stem in gt]
        experiment = TARGET / scene.name / f"{nominal}views"
        input_dir = experiment / "input"
        input_dir.mkdir(parents=True, exist_ok=True)
        manifest = {
            "scene": scene.name,
            "nominal_views": nominal,
            "available_gt_views": len(matched),
            "skipped_frames": [p.stem for p in source_images if p not in matched],
        }
        if not matched:
            (experiment / "subset_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
            continue
        for path in matched:
            destination = input_dir / path.name
            if not destination.exists():
                destination.symlink_to(path.resolve())

        matrices = np.stack([np.asarray(gt[path.stem]["transform_matrix"], float) @ CV_FLIP for path in matched])
        matrices = np.linalg.inv(matrices[0])[None] @ matrices
        centers = matrices[:, :3, 3]
        baseline = np.linalg.norm(centers[:, None] - centers[None, :], axis=-1).max()
        if len(matched) > 1 and baseline > 1e-8:
            matrices[:, :3, 3] /= baseline

        frames = []
        for index, (path, c2w) in enumerate(zip(matched, matrices)):
            width, height = Image.open(path).size
            sx, sy = width / tf["w"], height / tf["h"]
            fx, fy = tf["fl_x"] * sx, tf["fl_y"] * sy
            cx, cy = tf.get("cx", tf["w"] / 2) * sx, tf.get("cy", tf["h"] / 2) * sy
            side = min(width, height)
            cx = (cx - (width - side) / 2) * SIZE / side
            cy = (cy - (height - side) / 2) * SIZE / side
            fx, fy = fx * SIZE / side, fy * SIZE / side
            frames.append({
                "index": index, "name": path.stem, "c2w": c2w.tolist(),
                "intrinsics": [fx, fy, cx, cy],
                "K": [[fx, 0, cx], [0, fy, cy], [0, 0, 1]],
            })
        cameras = {
            "camera_source": "DL3DV transforms.json ground truth; missing sampled frames skipped",
            "nominal_views": nominal,
            "actual_views": len(matched),
            "frames": frames,
        }
        (experiment / "camera_priors.json").write_text(json.dumps(cameras, indent=2) + "\n")
        transforms_copy = experiment / "transforms.json"
        if not transforms_copy.exists():
            shutil.copy2(scene / "transforms.json", transforms_copy)
        (experiment / "subset_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        print(scene.name, nominal, len(matched))
