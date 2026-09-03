#!/usr/bin/env python3
"""Render QuerySplat full and opacity-filtered PLYs at predicted input cameras."""

import argparse
import json
import time
from pathlib import Path

import numpy as np

from render_gt_trajectory_videos import load_gaussians


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", type=int, required=True)
    parser.add_argument("--num-workers", type=int, required=True)
    parser.add_argument("--result-root", type=Path, action="append", required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    import torch
    from PIL import Image
    from zipsplat import Camera, Pose

    jobs = []
    variants = (
        ("full", "gaussians_opacity0.ply", "rendered_full_gs"),
        ("filtered", "gaussians_opacity0.05.ply", "rendered_filtered_gs"),
    )
    for root in args.result_root:
        for output in root.glob("[0-9][0-9]_*/[0-9]*views/*/output"):
            for variant, filename, directory in variants:
                ply = output / filename
                if not ply.is_file() and variant == "filtered":
                    ply = output / "gaussians.ply"
                if ply.is_file() and (output / "predicted_input_cameras.json").is_file():
                    jobs.append((variant, ply, output / directory))

    for variant, ply, destination in sorted(jobs)[args.worker :: args.num_workers]:
        done = destination / "render_manifest.json"
        if done.is_file() and not args.overwrite:
            continue
        payload = json.loads((ply.parent / "predicted_input_cameras.json").read_text())
        frames = sorted(payload["frames"], key=lambda frame: int(frame["index"]))
        matrices = np.asarray([frame["c2w"] for frame in frames], np.float32)
        intrinsics = np.asarray(
            [[frame["K"][0][0], frame["K"][1][1], frame["K"][0][2], frame["K"][1][2]] for frame in frames],
            np.float32,
        )
        scale = 128.0 / np.maximum((intrinsics[:, 2] + intrinsics[:, 3]) / 2, 1e-6)
        intrinsics *= scale[:, None]
        cameras = Camera.from_focal(
            torch.from_numpy(intrinsics[:, 0]), torch.from_numpy(intrinsics[:, 1]),
            w=256, h=256, cx=torch.from_numpy(intrinsics[:, 2]), cy=torch.from_numpy(intrinsics[:, 3]),
        )
        poses = Pose.from_4x4mat(torch.from_numpy(matrices))
        gaussians = load_gaussians(ply, "cuda")
        # Warm up the renderer so the recorded value does not include first-call
        # CUDA extension/kernel initialization for whichever variant runs first.
        gaussians.render(
            cameras.to("cuda"), poses.to("cuda"),
            backgrounds=torch.ones(len(frames), 3, device="cuda"),
        )
        torch.cuda.synchronize()
        started = time.perf_counter()
        images, _ = gaussians.render(
            cameras.to("cuda"), poses.to("cuda"),
            backgrounds=torch.ones(len(frames), 3, device="cuda"),
        )
        torch.cuda.synchronize()
        render_seconds = time.perf_counter() - started
        destination.mkdir(exist_ok=True)
        for frame, image in zip(frames, images):
            array = (image.detach().cpu().permute(1, 2, 0).clamp(0, 1).numpy() * 255).astype(np.uint8)
            Image.fromarray(array).save(destination / f"render_view{int(frame['index'])}.png")
        done.write_text(json.dumps({"variant": variant, "gaussian_source": ply.name, "views": len(frames), "render_seconds": render_seconds}, indent=2) + "\n")
        print(f"DONE {destination} variant={variant} views={len(frames)}", flush=True)


if __name__ == "__main__":
    main()
