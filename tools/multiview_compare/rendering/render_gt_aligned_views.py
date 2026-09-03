#!/usr/bin/env python3
"""Render existing QuerySplat/ZipSplat 3DGS at Sim(3)-aligned DL3DV GT input poses."""

import argparse
import json
from pathlib import Path

import numpy as np

from gaussian_io import align_gt_to_result, load_gaussians


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", type=int, required=True)
    parser.add_argument("--num-workers", type=int, required=True)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--result-root", type=Path, action="append", required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    import torch
    from PIL import Image
    from zipsplat import Camera, Pose

    jobs = []
    for result_root in args.result_root:
        for ply in result_root.glob("[0-9][0-9]_*/[0-9]*views/*/output/gaussians.ply"):
            relative = ply.parents[3].name + "/" + ply.parents[2].name
            priors = args.input_root / relative / "camera_priors.json"
            result_cameras = ply.parent / "predicted_input_cameras.json"
            if priors.is_file() and result_cameras.is_file():
                jobs.append((ply, priors, result_cameras))

    for ply, priors_path, result_camera_path in sorted(jobs)[args.worker :: args.num_workers]:
        destination = ply.parent / "rendered_gt_aligned"
        stats_path = ply.parent / "gt_aligned_render_stats.json"
        if stats_path.is_file() and not args.overwrite:
            continue
        destination.mkdir(exist_ok=True)
        gt_payload = json.loads(priors_path.read_text())
        gt_frames = gt_payload["frames"]
        result_payload = json.loads(result_camera_path.read_text())
        result_by_name = {f["name"]: f for f in result_payload["frames"]}
        matched_gt = [f for f in gt_frames if f["name"] in result_by_name]
        if len(matched_gt) < 3:
            print(f"SKIP {ply}: only {len(matched_gt)} matched GT cameras", flush=True)
            continue
        aligned, alignment = align_gt_to_result(gt_frames, result_camera_path)
        gt_index = {f["name"]: i for i, f in enumerate(gt_frames)}
        ordered = sorted(matched_gt, key=lambda f: int(result_by_name[f["name"]]["index"]))
        matrices = np.stack([aligned[gt_index[f["name"]]] for f in ordered]).astype(np.float32)
        intrinsics = np.asarray([f["intrinsics"] for f in ordered], np.float32)
        cameras = Camera.from_focal(
            torch.from_numpy(intrinsics[:, 0]), torch.from_numpy(intrinsics[:, 1]),
            w=256, h=256, cx=torch.from_numpy(intrinsics[:, 2]), cy=torch.from_numpy(intrinsics[:, 3]),
        )
        poses = Pose.from_4x4mat(torch.from_numpy(matrices))
        gaussians = load_gaussians(ply, "cuda")
        images, _ = gaussians.render(
            cameras.to("cuda"), poses.to("cuda"),
            backgrounds=torch.ones(len(ordered), 3, device="cuda"),
        )
        psnr = []
        for frame, image in zip(ordered, images):
            index = int(result_by_name[frame["name"]]["index"])
            array = (image.detach().cpu().permute(1, 2, 0).clamp(0, 1).numpy() * 255).astype(np.uint8)
            Image.fromarray(array).save(destination / f"render_view{index}.png")
            candidates = list((ply.parent / "input_frames").glob(f"view{index}_*"))
            if candidates:
                reference_image = Image.open(candidates[0]).convert("RGB")
                if reference_image.size != (256, 256):
                    reference_image = reference_image.resize((256, 256), Image.Resampling.LANCZOS)
                reference = np.asarray(reference_image, np.float32) / 255
                prediction = array.astype(np.float32) / 255
                mse = float(np.mean((prediction - reference) ** 2))
                psnr.append(-10 * np.log10(max(mse, 1e-10)))
        report = {
            **alignment,
            "rendered_views": len(ordered),
            "render_directory": str(destination),
            "per_view_psnr": psnr,
            "mean_psnr": float(np.mean(psnr)) if psnr else None,
        }
        stats_path.write_text(json.dumps(report, indent=2) + "\n")
        print(
            f"DONE {destination} views={len(ordered)} scale={alignment['scale']:.6g} "
            f"rmse={alignment['position_rmse']:.3g}", flush=True
        )


if __name__ == "__main__":
    main()
