#!/usr/bin/env python3
"""Render 3DGS videos along interpolated available ground-truth camera paths."""

import argparse
import json
from pathlib import Path

import numpy as np


def fit_sim3(source: np.ndarray, target: np.ndarray):
    """Least-squares similarity mapping target ~= scale * rotation @ source + translation."""
    if len(source) < 3:
        raise ValueError("At least three matched cameras are required for Sim(3) alignment")
    source_mean, target_mean = source.mean(0), target.mean(0)
    source_zero, target_zero = source - source_mean, target - target_mean
    covariance = target_zero.T @ source_zero / len(source)
    u, singular, vt = np.linalg.svd(covariance)
    sign = np.ones(3)
    if np.linalg.det(u @ vt) < 0:
        sign[-1] = -1
    rotation = u @ np.diag(sign) @ vt
    variance = np.mean(np.sum(source_zero * source_zero, axis=1))
    if variance < 1e-12:
        raise ValueError("Ground-truth camera centers have insufficient baseline")
    scale = float(np.sum(singular * sign) / variance)
    translation = target_mean - scale * (rotation @ source_mean)
    return scale, rotation, translation


def align_gt_to_result(gt_frames, result_camera_path: Path):
    """Map GT c2w poses into the coordinate system in which the result PLY was built."""
    payload = json.loads(result_camera_path.read_text())
    result_by_name = {frame["name"]: frame for frame in payload["frames"]}
    matched = [(frame, result_by_name[frame["name"]]) for frame in gt_frames if frame["name"] in result_by_name]
    gt_centers = np.stack([np.asarray(a["c2w"], float)[:3, 3] for a, _ in matched])
    result_centers = np.stack([np.asarray(b["c2w"], float)[:3, 3] for _, b in matched])
    scale, rotation, translation = fit_sim3(gt_centers, result_centers)
    matrices = np.asarray([frame["c2w"] for frame in gt_frames], np.float32)
    matrices[:, :3, 3] = (scale * (rotation @ matrices[:, :3, 3].T)).T + translation
    matrices[:, :3, :3] = rotation @ matrices[:, :3, :3]
    fitted = (scale * (rotation @ gt_centers.T)).T + translation
    residuals = np.linalg.norm(fitted - result_centers, axis=1)
    report = {
        "mapping": "ground_truth_to_result_3dgs_coordinate_system",
        "matched_cameras": len(matched),
        "scale": scale,
        "rotation": rotation.tolist(),
        "translation": translation.tolist(),
        "position_rmse": float(np.sqrt(np.mean(residuals ** 2))),
        "position_max_error": float(residuals.max()),
        "result_cameras": str(result_camera_path),
    }
    return matrices, report


def load_gaussians(path, device):
    import torch
    from plyfile import PlyData
    from zipsplat import Gaussians
    v = PlyData.read(path)["vertex"].data
    xyz = np.stack([v[k] for k in ("x", "y", "z")], -1)
    scales = np.exp(np.stack([v[k] for k in ("scale_0", "scale_1", "scale_2")], -1))
    quats = np.stack([v[f"rot_{i}"] for i in range(4)], -1)
    opacity = 1 / (1 + np.exp(-np.asarray(v["opacity"])))
    dc = np.stack([v[f"f_dc_{i}"] for i in range(3)], -1)[:, None]
    names = sorted((n for n in v.dtype.names if n.startswith("f_rest_")), key=lambda n: int(n.rsplit("_", 1)[1]))
    if names:
        rest = np.stack([v[n] for n in names], -1).reshape(len(v), 3, -1).transpose(0, 2, 1)
        sh = np.concatenate([dc, rest], 1)
    else:
        sh = dc
    tensor = lambda x: torch.from_numpy(x).float().to(device)
    return Gaussians.from_parameters(tensor(xyz), tensor(scales), tensor(quats), tensor(opacity), tensor(sh))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", type=int, required=True)
    parser.add_argument("--num-workers", type=int, required=True)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--result-root", type=Path, action="append", required=True)
    parser.add_argument("--num-frames", type=int, default=240)
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    import torch
    from scipy.spatial.transform import Rotation, Slerp
    from zipsplat import Camera, Pose, viz

    jobs = []
    for result_root in args.result_root:
        for ply in result_root.glob("[0-9][0-9]_*/[0-9]*views/*/output/gaussians.ply"):
            relative = ply.parents[3].name + "/" + ply.parents[2].name
            camera_json = args.input_root / relative / "camera_priors.json"
            if camera_json.is_file():
                jobs.append((ply, camera_json))
    for ply, camera_json in sorted(jobs)[args.worker :: args.num_workers]:
        output = ply.parent / "video_gt_trajectory.mp4"
        if output.is_file() and not args.overwrite:
            continue
        frames = json.loads(camera_json.read_text())["frames"]
        if len(frames) < 2:
            print(f"SKIP {ply}: only {len(frames)} GT camera", flush=True)
            continue
        result_cameras = ply.parent / "predicted_input_cameras.json"
        if not result_cameras.is_file():
            print(f"SKIP {ply}: missing {result_cameras.name} for GT-to-scene alignment", flush=True)
            continue
        try:
            mats, alignment = align_gt_to_result(frames, result_cameras)
        except ValueError as error:
            print(f"SKIP {ply}: {error}", flush=True)
            continue
        (ply.parent / "gt_trajectory_alignment.json").write_text(json.dumps(alignment, indent=2) + "\n")
        intr = np.asarray([f["intrinsics"] for f in frames], np.float32)
        key = np.arange(len(frames), dtype=float)
        sample = np.linspace(0, len(frames) - 1, args.num_frames)
        positions = np.stack([np.interp(sample, key, mats[:, i, 3]) for i in range(3)], -1)
        rotations = Slerp(key, Rotation.from_matrix(mats[:, :3, :3]))(sample).as_matrix().astype(np.float32)
        intrinsics = np.stack([np.interp(sample, key, intr[:, i]) for i in range(4)], -1)
        c2w = np.tile(np.eye(4, dtype=np.float32), (len(sample), 1, 1))
        c2w[:, :3, :3], c2w[:, :3, 3] = rotations, positions
        poses = Pose.from_4x4mat(torch.from_numpy(c2w))
        cameras = Camera.from_focal(
            torch.from_numpy(intrinsics[:, 0]), torch.from_numpy(intrinsics[:, 1]),
            w=256, h=256, cx=torch.from_numpy(intrinsics[:, 2]), cy=torch.from_numpy(intrinsics[:, 3]),
        )
        g = load_gaussians(ply, "cuda")
        images = viz.render_video(g, cameras, poses, chunk=8)
        viz.save_video(images, output, fps=args.fps)
        print(
            f"DONE {output} sim3_scale={alignment['scale']:.6g} "
            f"align_rmse={alignment['position_rmse']:.3g}", flush=True
        )


if __name__ == "__main__":
    main()
