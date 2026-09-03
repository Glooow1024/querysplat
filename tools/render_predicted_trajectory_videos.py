#!/usr/bin/env python3
"""Render full interpolation videos using only each reconstruction's estimated input poses."""

import argparse
import json
from pathlib import Path

import numpy as np

from render_full_trajectory_videos import estimated_intrinsics, interpolate
from render_gt_trajectory_videos import load_gaussians


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", type=int, required=True)
    parser.add_argument("--num-workers", type=int, required=True)
    parser.add_argument("--result-root", type=Path, action="append", required=True)
    parser.add_argument("--num-frames", type=int, default=240)
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--full-gaussians", action="store_true")
    args = parser.parse_args()

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
        output = ply.parent / ("video_predicted_trajectory_full_gs.mp4" if args.full_gaussians else "video_predicted_trajectory.mp4")
        if output.is_file() and not args.overwrite:
            continue
        payload = json.loads((ply.parent / "predicted_input_cameras.json").read_text())
        frames = payload["frames"]
        matrices = np.asarray([frame["c2w"] for frame in frames], np.float32)
        intrinsics = estimated_intrinsics(frames)
        matrices, intrinsics = interpolate(matrices, intrinsics, args.num_frames)
        poses = Pose.from_4x4mat(torch.from_numpy(matrices))
        cameras = Camera.from_focal(
            torch.from_numpy(intrinsics[:, 0]), torch.from_numpy(intrinsics[:, 1]),
            w=256, h=256, cx=torch.from_numpy(intrinsics[:, 2]), cy=torch.from_numpy(intrinsics[:, 3]),
        )
        gaussians = load_gaussians(ply, "cuda")
        images = viz.render_video(gaussians, cameras, poses, chunk=8)
        viz.save_video(images, output, fps=args.fps)
        (ply.parent / "predicted_trajectory_manifest.json").write_text(json.dumps({
            "trajectory_source": "estimated_input_poses",
            "source_pose_count": len(frames), "video_frames": args.num_frames, "fps": args.fps,
            "gaussian_source": ply.name,
        }, indent=2) + "\n")
        print(f"DONE {output} poses={len(frames)}", flush=True)


if __name__ == "__main__":
    main()
