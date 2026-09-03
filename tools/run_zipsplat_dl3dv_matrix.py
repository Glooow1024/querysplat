#!/usr/bin/env python3
"""Run ZipSplat on the exact QuerySplat random10 view-matrix inputs."""

import argparse
import gc
import glob
import json
import os
import time
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", type=int, required=True)
    parser.add_argument("--num-workers", type=int, required=True)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--compression", type=float, default=1.0)
    parser.add_argument("--video-frames", type=int, default=48)
    args = parser.parse_args()

    import torch
    from zipsplat import ZipSplat, load_image, viz

    jobs = []
    for scene_dir in sorted(args.input_root.glob("[0-9][0-9]_*")):
        if not scene_dir.is_dir():
            continue
        for views in (4, 8, 12, 16):
            experiment = scene_dir / f"{views}views"
            images = sorted((experiment / "input").glob("*.png"))
            if len(images) != views:
                raise RuntimeError(f"{experiment}: expected {views} PNGs, found {len(images)}")
            jobs.append((scene_dir.name, views, experiment, images))

    assigned = jobs[args.worker :: args.num_workers]
    print(f"worker={args.worker} jobs={len(assigned)} device={torch.cuda.get_device_name(0)}", flush=True)
    load_start = time.perf_counter()
    model = ZipSplat(weights="zipsplat").cuda().eval()
    model_load_seconds = time.perf_counter() - load_start

    for scene_name, views, experiment, image_paths in assigned:
        output_dir = args.output_root / scene_name / f"{views}views"
        output_dir.mkdir(parents=True, exist_ok=True)
        metrics_path = output_dir / "run_stats.json"
        if metrics_path.exists() and (output_dir / "scene.ply").exists():
            print(f"SKIP {scene_name}/{views}views", flush=True)
            continue

        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        images = [load_image(str(path)) for path in image_paths]
        infer_start = time.perf_counter()
        gaussians = model(images, compression=args.compression)[0]
        torch.cuda.synchronize()
        infer_seconds = time.perf_counter() - infer_start
        gaussians.save_ply(output_dir / "scene.ply")

        video_seconds = None
        if args.video_frames > 0:
            video_start = time.perf_counter()
            viz.turntable(
                gaussians,
                output_dir / "turntable.mp4",
                render_size=384,
                num_frames=args.video_frames,
                fps=24,
                sweep_deg=None,
                chunk=8,
            )
            torch.cuda.synchronize()
            video_seconds = time.perf_counter() - video_start

        stats = {
            "scene": scene_name,
            "views": views,
            "input_experiment": str(experiment),
            "input_images": [str(path) for path in image_paths],
            "input_image_targets": [str(path.resolve()) for path in image_paths],
            "compression": args.compression,
            "num_gaussians": gaussians.num_gaussians,
            "model_load_seconds": model_load_seconds,
            "inference_seconds": infer_seconds,
            "video_seconds": video_seconds,
            "total_seconds": time.perf_counter() - started,
            "peak_cuda_allocated_gib": torch.cuda.max_memory_allocated() / (1024**3),
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0),
        }
        metrics_path.write_text(json.dumps(stats, indent=2) + "\n")
        print(
            f"DONE {scene_name}/{views}views inference={infer_seconds:.3f}s "
            f"gaussians={gaussians.num_gaussians} peak={stats['peak_cuda_allocated_gib']:.2f}GiB",
            flush=True,
        )
        del images, gaussians
        gc.collect()


if __name__ == "__main__":
    main()
