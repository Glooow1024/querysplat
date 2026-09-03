#!/usr/bin/env python3
"""ZipSplat DL3DV inference with explicitly labelled GT camera priors."""

import argparse
import gc
import json
import math
import struct
import time
from pathlib import Path

import numpy as np


def find_transforms(image: Path) -> Path:
    # Prefer the full trajectory generated beside the benchmark selection.
    # The source Nerfstudio transforms may contain only the two held-out views.
    for candidate_image in (image, image.resolve()):
        for parent in candidate_image.parents:
            candidate = parent / "transforms.json"
            if candidate.is_file():
                return candidate
    raise FileNotFoundError(f"No transforms.json above {image.resolve()}")


def frame_name(path: Path) -> str:
    return path.stem


def qvec_to_rot(q):
    w, x, y, z = q
    return np.array([
        [1 - 2 * y * y - 2 * z * z, 2 * x * y - 2 * w * z, 2 * x * z + 2 * w * y],
        [2 * x * y + 2 * w * z, 1 - 2 * x * x - 2 * z * z, 2 * y * z - 2 * w * x],
        [2 * x * z - 2 * w * y, 2 * y * z + 2 * w * x, 1 - 2 * x * x - 2 * y * y],
    ], dtype=np.float32)


def read_colmap_c2w(image_path: Path) -> dict[str, np.ndarray]:
    model = next(
        (parent / "colmap" / "sparse" / "0" / "images.bin" for parent in image_path.resolve().parents
         if (parent / "colmap" / "sparse" / "0" / "images.bin").is_file()),
        None,
    )
    if model is None:
        raise FileNotFoundError(f"No COLMAP images.bin above {image_path.resolve()}")
    result = {}
    with model.open("rb") as stream:
        count = struct.unpack("<Q", stream.read(8))[0]
        for _ in range(count):
            _, *values = struct.unpack("<i7di", stream.read(64))
            q, t = values[:4], np.asarray(values[4:7], np.float32)
            name_bytes = bytearray()
            while (char := stream.read(1)) != b"\0":
                name_bytes.extend(char)
            name = Path(name_bytes.decode()).stem
            npoints = struct.unpack("<Q", stream.read(8))[0]
            stream.seek(24 * npoints, 1)
            w2c_r = qvec_to_rot(q)
            c2w = np.eye(4, dtype=np.float32)
            c2w[:3, :3] = w2c_r.T
            c2w[:3, 3] = -(w2c_r.T @ t)
            result[name] = c2w
    return result


def save_pointcloud(gaussians, path: Path, opacity_threshold: float = 0.005) -> int:
    from plyfile import PlyData, PlyElement

    xyz = gaussians.means.detach().float().cpu().numpy()
    rgb = (gaussians.rgb.detach().float().clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
    opacity = gaussians.opacities.detach().float().cpu().numpy()
    keep = opacity >= opacity_threshold
    xyz, rgb = xyz[keep], rgb[keep]
    dtype = [("x", "f4"), ("y", "f4"), ("z", "f4"), ("red", "u1"), ("green", "u1"), ("blue", "u1")]
    vertices = np.empty(len(xyz), dtype=dtype)
    vertices["x"], vertices["y"], vertices["z"] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    vertices["red"], vertices["green"], vertices["blue"] = rgb[:, 0], rgb[:, 1], rgb[:, 2]
    PlyData([PlyElement.describe(vertices, "vertex")]).write(path)
    return len(vertices)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", type=int, required=True)
    parser.add_argument("--num-workers", type=int, required=True)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--external-priors", action="store_true")
    parser.add_argument("--input-method")
    parser.add_argument("--output-method")
    args = parser.parse_args()

    import torch
    from PIL import Image
    from zipsplat import Camera, Pose, ZipSplat, load_image, viz
    from zipsplat.utils import IMAGE_SIZE, to_square

    jobs = []
    for scene in sorted(args.input_root.glob("[0-9][0-9]_*")):
        if scene.is_dir():
            for views in (4, 8, 12, 14, 16):
                experiment = scene / f"{views}views"
                if args.input_method:
                    experiment = experiment / args.input_method
                images = sorted((experiment / "input").glob("*.png"))
                if images and (args.external_priors or len(images) == views):
                    jobs.append((scene.name, views, images))

    model_load_start = time.perf_counter()
    model = ZipSplat(weights="zipsplat").cuda().eval()
    model_load_seconds = time.perf_counter() - model_load_start
    cv_flip = np.diag([1.0, -1.0, -1.0, 1.0]).astype(np.float32)

    for scene_name, views, image_paths in jobs[args.worker :: args.num_workers]:
        out_base = args.output_root / scene_name / f"{views}views"
        if args.output_method:
            out_base = out_base / args.output_method
        out = out_base / "output"
        if (out / "run_stats.json").is_file() and (out / "gaussians.ply").is_file():
            print(f"SKIP {scene_name}/{views}views", flush=True)
            continue
        out.mkdir(parents=True, exist_ok=True)
        (out / "input_frames").mkdir(exist_ok=True)
        (out / "rendered").mkdir(exist_ok=True)

        experiment = image_paths[0].parent.parent
        tf_path = find_transforms(image_paths[0])
        tf = json.loads(tf_path.read_text())
        if args.external_priors:
            camera_payload = json.loads((experiment / "camera_priors.json").read_text())
            query_frames = camera_payload["frames"]
            raw = [load_image(str(p)) for p in image_paths]
            camera_mode = "DL3DV_ground_truth_cameras_missing_frames_skipped"
            camera_warning = "Uses available DL3DV ground-truth cameras; missing sampled frames were skipped."
        else:
            query_output = experiment / "output"
            camera_payload = json.loads((query_output / "predicted_input_cameras.json").read_text())
            query_frames = camera_payload["frames"]
            query_inputs = [query_output / f["source"] for f in query_frames]
            raw = [load_image(str(p)) for p in query_inputs]
            camera_mode = "shared_querysplat_predicted_cameras_max_pairwise_normalized"
            camera_warning = "Uses QuerySplat-predicted cameras as shared priors; compares reconstruction/rendering, not independent pose estimation."
        if [f["name"] for f in query_frames] != [frame_name(p) for p in image_paths]:
            raise RuntimeError(f"Camera/image ordering mismatch in {experiment}")
        raw_tensor = torch.stack(raw)
        h, w = raw_tensor.shape[-2:]

        c2w = np.stack([np.asarray(f["c2w"], np.float32) for f in query_frames])
        pose_source = camera_payload.get("camera_source", "camera_priors.json")
        poses = Pose.from_4x4mat(torch.from_numpy(c2w))

        Ks = torch.tensor(np.stack([f["K"] for f in query_frames]), dtype=torch.float32)
        cameras = Camera.from_K(Ks, w=w, h=h)
        side = min(h, w)
        render_cameras = cameras.crop((side - w, side - h)).scale(IMAGE_SIZE / side)
        prepared = to_square(raw_tensor)

        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        infer_start = time.perf_counter()
        gaussians = model(raw_tensor, cameras=cameras, poses=poses, use_priors=True)[0]
        torch.cuda.synchronize()
        infer_seconds = time.perf_counter() - infer_start
        render_start = time.perf_counter()
        actual_views = len(image_paths)
        rendered, _ = gaussians.render(render_cameras, poses, backgrounds=torch.ones(actual_views, 3, device="cuda"))
        torch.cuda.synchronize()
        render_seconds = time.perf_counter() - render_start

        for i, (path, source, prediction) in enumerate(zip(image_paths, prepared, rendered)):
            name = frame_name(path)
            Image.fromarray((source.permute(1, 2, 0).clamp(0, 1).numpy() * 255).astype(np.uint8)).save(
                out / "input_frames" / f"view{i}_{name}_preprocessed.png"
            )
            Image.fromarray((prediction.detach().cpu().permute(1, 2, 0).clamp(0, 1).numpy() * 255).astype(np.uint8)).save(
                out / "rendered" / f"render_view{i}.png"
            )
        gaussians.save_ply(out / "gaussians.ply")
        point_count = save_pointcloud(gaussians, out / "pointcloud.ply")
        viz.turntable(gaussians, out / "video.mp4", render_size=384, num_frames=48, fps=24, sweep_deg=None, chunk=8)

        camera_payload = {
            "camera_source": pose_source,
            "frames": [
                {
                    "index": i,
                    "name": frame_name(path),
                    "source": f"input_frames/view{i}_{frame_name(path)}_preprocessed.png",
                    "c2w": c2w[i].tolist(),
                    "K": render_cameras.K[i].tolist(),
                    "intrinsics": [float(x) for x in (*render_cameras.f[i].tolist(), *render_cameras.c[i].tolist())],
                }
                for i, path in enumerate(image_paths)
            ],
        }
        (out / "predicted_input_cameras.json").write_text(json.dumps(camera_payload, indent=2) + "\n")
        mse = torch.mean((rendered.detach().cpu() - prepared) ** 2, dim=(1, 2, 3))
        psnr = (-10 * torch.log10(mse.clamp_min(1e-10))).tolist()
        stats = {
            "algorithm": "zipsplat",
            "camera_mode": camera_mode,
            "pose_source": pose_source,
            "warning": camera_warning,
            "scene": scene_name,
            "nominal_views": views,
            "views": actual_views,
            "num_gaussians": gaussians.num_gaussians,
            "pointcloud_count": point_count,
            "model_load_seconds": model_load_seconds,
            "inference_seconds": infer_seconds,
            "render_seconds": render_seconds,
            "total_seconds": time.perf_counter() - started,
            "peak_cuda_allocated_gib": torch.cuda.max_memory_allocated() / 1024**3,
            "input_view_psnr": psnr,
            "mean_input_view_psnr": float(np.mean(psnr)),
            "transforms": str(tf_path),
        }
        (out / "run_stats.json").write_text(json.dumps(stats, indent=2) + "\n")
        (out / "result_manifest.json").write_text(json.dumps({"schema": 1, **stats}, indent=2) + "\n")
        print(f"DONE {scene_name}/{views}views psnr={np.mean(psnr):.2f} infer={infer_seconds:.2f}s", flush=True)
        del raw, raw_tensor, prepared, rendered, gaussians
        gc.collect()


if __name__ == "__main__":
    main()
