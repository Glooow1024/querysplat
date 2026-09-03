#!/usr/bin/env python3
"""Render a full-path video, preferring aligned full GT and falling back to estimated poses."""

import argparse
import json
import re
from pathlib import Path

import numpy as np

from gaussian_io import fit_sim3, load_gaussians

CV_FLIP = np.diag([1.0, -1.0, -1.0, 1.0])


def name_of(value):
    return Path(value).stem


def order_key(frame):
    numbers = re.findall(r"\d+", name_of(frame.get("file_path", frame.get("name", ""))))
    return int(numbers[-1]) if numbers else 10**12


def interpolate(matrices, intrinsics, count):
    from scipy.spatial.transform import Rotation, Slerp
    key = np.arange(len(matrices), dtype=float)
    sample = np.linspace(0, len(matrices) - 1, count)
    positions = np.stack([np.interp(sample, key, matrices[:, i, 3]) for i in range(3)], -1)
    rotations = Slerp(key, Rotation.from_matrix(matrices[:, :3, :3]))(sample).as_matrix()
    intr = np.stack([np.interp(sample, key, intrinsics[:, i]) for i in range(4)], -1)
    result = np.tile(np.eye(4), (count, 1, 1))
    result[:, :3, :3], result[:, :3, 3] = rotations, positions
    return result.astype(np.float32), intr.astype(np.float32)


def gt_intrinsics(tf, count):
    w, h = float(tf["w"]), float(tf["h"])
    side = min(w, h)
    fx, fy = float(tf["fl_x"]), float(tf["fl_y"])
    cx, cy = float(tf.get("cx", w / 2)), float(tf.get("cy", h / 2))
    cx = (cx - (w - side) / 2) * 256 / side
    cy = (cy - (h - side) / 2) * 256 / side
    return np.tile([fx * 256 / side, fy * 256 / side, cx, cy], (count, 1))


def estimated_intrinsics(frames):
    values = []
    for frame in frames:
        K = np.asarray(frame["K"], float)
        factor = 128.0 / max((K[0, 2] + K[1, 2]) / 2, 1e-6)
        values.append([K[0, 0] * factor, K[1, 1] * factor, K[0, 2] * factor, K[1, 2] * factor])
    return np.asarray(values)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--worker", type=int, required=True)
    p.add_argument("--num-workers", type=int, required=True)
    p.add_argument("--reference-root", type=Path, required=True)
    p.add_argument("--result-root", type=Path, action="append", required=True)
    p.add_argument("--num-frames", type=int, default=240)
    p.add_argument("--fps", type=int, default=24)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--full-gaussians", action="store_true")
    args = p.parse_args()

    import torch
    from zipsplat import Camera, Pose, viz

    jobs = []
    for root in args.result_root:
        for output_dir in root.glob("[0-9][0-9]_*/[0-9]*views/*/output"):
            if args.full_gaussians:
                ply = output_dir / "gaussians_opacity0.ply"
            else:
                ply = output_dir / "gaussians.ply"
                if not ply.is_file():
                    ply = output_dir / "gaussians_opacity0.05.ply"
            if ply.is_file():
                jobs.append(ply)
    for ply in sorted(jobs)[args.worker :: args.num_workers]:
        output = ply.parent / ("video_full_trajectory_full_gs.mp4" if args.full_gaussians else "video_full_trajectory.mp4")
        report_path = ply.parent / ("full_trajectory_full_gs_manifest.json" if args.full_gaussians else "full_trajectory_manifest.json")
        if output.is_file() and not args.overwrite:
            continue
        scene = ply.parents[3].name
        result_payload = json.loads((ply.parent / "predicted_input_cameras.json").read_text())
        result_frames = result_payload["frames"]
        result_by_name = {f["name"]: f for f in result_frames}
        tf_path = args.reference_root / scene / "transforms.json"
        source = "estimated_input_poses"
        alignment = None
        matrices = np.asarray([f["c2w"] for f in result_frames], float)
        intrinsics = estimated_intrinsics(result_frames)

        if tf_path.is_file():
            tf = json.loads(tf_path.read_text())
            full_frames = sorted(tf.get("frames", []), key=order_key)
            gt_by_name = {name_of(f["file_path"]): np.asarray(f["transform_matrix"], float) @ CV_FLIP for f in full_frames}
            names = sorted(set(gt_by_name) & set(result_by_name))
            if len(names) >= 3:
                gt_centers = np.stack([gt_by_name[n][:3, 3] for n in names])
                result_centers = np.stack([np.asarray(result_by_name[n]["c2w"])[:3, 3] for n in names])
                scale, rotation, translation = fit_sim3(gt_centers, result_centers)
                matrices = np.stack([gt_by_name[name_of(f["file_path"])] for f in full_frames])
                matrices[:, :3, 3] = (scale * (rotation @ matrices[:, :3, 3].T)).T + translation
                matrices[:, :3, :3] = rotation @ matrices[:, :3, :3]
                intrinsics = gt_intrinsics(tf, len(matrices))
                fitted = (scale * (rotation @ gt_centers.T)).T + translation
                residual = np.linalg.norm(fitted - result_centers, axis=1)
                alignment = {"scale": scale, "rotation": rotation.tolist(), "translation": translation.tolist(),
                             "rmse": float(np.sqrt(np.mean(residual ** 2))), "matched_frames": len(names)}
                source = "aligned_full_ground_truth"

        matrices, intrinsics = interpolate(matrices, intrinsics, args.num_frames)
        poses = Pose.from_4x4mat(torch.from_numpy(matrices))
        cameras = Camera.from_focal(torch.from_numpy(intrinsics[:, 0]), torch.from_numpy(intrinsics[:, 1]),
                                    w=256, h=256, cx=torch.from_numpy(intrinsics[:, 2]), cy=torch.from_numpy(intrinsics[:, 3]))
        gaussians = load_gaussians(ply, "cuda")
        images = viz.render_video(gaussians, cameras, poses, chunk=8)
        viz.save_video(images, output, fps=args.fps)
        report = {"trajectory_source": source, "source_pose_count": int(len(result_frames) if source.startswith("estimated") else len(full_frames)),
                  "video_frames": args.num_frames, "fps": args.fps, "alignment": alignment, "gaussian_source": ply.name}
        report_path.write_text(json.dumps(report, indent=2) + "\n")
        print(f"DONE {output} source={source} poses={report['source_pose_count']}", flush=True)


if __name__ == "__main__":
    main()
