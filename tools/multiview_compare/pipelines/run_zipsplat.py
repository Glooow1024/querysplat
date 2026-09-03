#!/usr/bin/env python3
"""ZipSplat DL3DV inference with shared, GT, or no camera priors."""

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
    parser.add_argument(
        "--no-priors",
        action="store_true",
        help="Reconstruct without camera/intrinsics input, then register the shared trajectory post hoc.",
    )
    parser.add_argument("--input-method")
    parser.add_argument("--output-method")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--pose-refinement-steps", type=int, default=200)
    args = parser.parse_args()
    if args.external_priors and args.no_priors:
        parser.error("--external-priors and --no-priors are mutually exclusive")

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
    pose_lpips = None
    cv_flip = np.diag([1.0, -1.0, -1.0, 1.0]).astype(np.float32)

    for scene_name, views, image_paths in jobs[args.worker :: args.num_workers]:
        out_base = args.output_root / scene_name / f"{views}views"
        if args.output_method:
            out_base = out_base / args.output_method
        out = out_base / "output"
        if not args.overwrite and (out / "run_stats.json").is_file() and (out / "gaussians.ply").is_file():
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
            if args.no_priors:
                raw = [load_image(str(p)) for p in image_paths]
                camera_mode = "pose_free_reconstruction_posthoc_input_only_registration"
                camera_warning = (
                    "ZipSplat reconstruction receives images only. Following the official pose-free benchmark, "
                    "DL3DV intrinsics and normalized GT poses initialize post-hoc per-view pose refinement."
                )
            else:
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
        if args.no_priors:
            pose_source = "DL3DV_GT_initialization_then_official_per_view_photometric_LPIPS_refinement"
        if args.no_priors:
            # Official pose-free evaluation uses dataset cameras for post-hoc rendering;
            # intrinsics are not passed into the reconstruction network.
            sx, sy = w / float(tf["w"]), h / float(tf["h"])
            gt_K = np.array(
                [[tf["fl_x"] * sx, 0.0, tf.get("cx", tf["w"] / 2) * sx],
                 [0.0, tf["fl_y"] * sy, tf.get("cy", tf["h"] / 2) * sy],
                 [0.0, 0.0, 1.0]],
                dtype=np.float32,
            )
            Ks = torch.tensor(np.repeat(gt_K[None], len(query_frames), axis=0))
        else:
            Ks = torch.tensor(np.stack([f["K"] for f in query_frames]), dtype=torch.float32)
        cameras = Camera.from_K(Ks, w=w, h=h)
        side = min(h, w)
        render_cameras = cameras.crop((side - w, side - h)).scale(IMAGE_SIZE / side)
        prepared = to_square(raw_tensor)
        actual_views = len(image_paths)

        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        infer_start = time.perf_counter()
        if args.no_priors:
            gaussians = model(raw_tensor)[0]
        else:
            poses = Pose.from_4x4mat(torch.from_numpy(c2w))
            gaussians = model(raw_tensor, cameras=cameras, poses=poses, use_priors=True)[0]
        torch.cuda.synchronize()
        infer_seconds = time.perf_counter() - infer_start

        registration_seconds = 0.0
        registration_scale = None
        registration_loss = None
        if args.no_priors:
            registration_start = time.perf_counter()
            axis_flip = np.diag([1.0, -1.0, -1.0, 1.0]).astype(np.float32)
            gt_by_name = {
                Path(frame["file_path"]).stem: np.asarray(frame["transform_matrix"], np.float32) @ axis_flip
                for frame in tf["frames"]
            }
            missing_gt = [frame["name"] for frame in query_frames if frame["name"] not in gt_by_name]
            if missing_gt:
                # Some converted DL3DV transforms files only list an evaluation
                # subset.  Use one coherent COLMAP coordinate system for the
                # entire selected set rather than mixing transformed and raw poses.
                colmap_c2w = read_colmap_c2w(image_paths[0])
                missing_colmap = [frame["name"] for frame in query_frames if frame["name"] not in colmap_c2w]
                if missing_colmap:
                    raise RuntimeError(f"Missing both transforms and COLMAP GT poses for: {missing_colmap}")
                gt_c2w = np.stack([colmap_c2w[frame["name"]] for frame in query_frames])
                pose_source = "colmap_opencv_relative_first_context"
            else:
                gt_c2w = np.stack([gt_by_name[frame["name"]] for frame in query_frames])
                pose_source = "transforms_opencv_relative_first_context"
            first_inv = np.linalg.inv(gt_c2w[0])
            relative_c2w = first_inv[None] @ gt_c2w
            pose_conventions = {
                pose_source: relative_c2w,
            }
            target = prepared.to(device="cuda")

            def score(base: np.ndarray, scale: float) -> float:
                candidate = base.copy()
                candidate[:, :3, 3] *= scale
                candidate_poses = Pose.from_4x4mat(torch.from_numpy(candidate.astype(np.float32)))
                with torch.no_grad():
                    candidate_render, _ = gaussians.render(
                        render_cameras,
                        candidate_poses,
                        backgrounds=torch.ones(actual_views, 3, device="cuda"),
                    )
                    return float(torch.mean((candidate_render - target) ** 2).item())

            coarse = np.geomspace(0.03, 30.0, 17)
            coarse_scores = [
                (name, float(scale), score(base, float(scale)))
                for name, base in pose_conventions.items()
                for scale in coarse
            ]
            best_convention, best_scale, best_loss = min(coarse_scores, key=lambda item: item[2])
            fine = np.geomspace(best_scale / 1.6, best_scale * 1.6, 11)
            fine_scores = [
                (best_convention, float(scale), score(pose_conventions[best_convention], float(scale)))
                for scale in fine
            ]
            registration_convention, registration_scale, registration_loss = min(
                coarse_scores + fine_scores, key=lambda item: item[2]
            )
            base = torch.from_numpy(pose_conventions[registration_convention].astype(np.float32)).to("cuda")
            base_rotation = base[:, :3, :3]
            base_translation = base[:, :3, 3]
            rotation_delta = torch.nn.Parameter(torch.tensor([1e-6, 0.0, 0.0], device="cuda"))
            translation_delta = torch.nn.Parameter(torch.zeros(3, device="cuda"))
            log_scale = torch.nn.Parameter(torch.tensor(math.log(registration_scale), device="cuda"))
            optimizer = torch.optim.Adam(
                [
                    {"params": [rotation_delta, translation_delta], "lr": 0.01},
                    {"params": [log_scale], "lr": 0.02},
                ]
            )
            best = (registration_loss, rotation_delta.detach().clone(), translation_delta.detach().clone(), log_scale.detach().clone())

            def rotation_matrix(vector: torch.Tensor) -> torch.Tensor:
                x, y, z = vector.unbind(-1)
                zero = torch.zeros_like(x)
                skew = torch.stack((zero, -z, y, z, zero, -x, -y, x, zero), -1).reshape(*vector.shape[:-1], 3, 3)
                theta = torch.linalg.vector_norm(vector, dim=-1).clamp_min(1e-8)
                a = (torch.sin(theta) / theta)[..., None, None]
                b = ((1.0 - torch.cos(theta)) / (theta * theta))[..., None, None]
                identity = torch.eye(3, device=vector.device).expand(*vector.shape[:-1], 3, 3)
                return identity + a * skew + b * (skew @ skew)

            for _ in range(120):
                optimizer.zero_grad()
                global_rotation = rotation_matrix(rotation_delta)
                registered_rotation = global_rotation[None] @ base_rotation
                registered_translation = (
                    torch.exp(log_scale) * (global_rotation[None] @ base_translation[..., None]).squeeze(-1)
                    + translation_delta
                )
                candidate_poses = Pose.from_Rt(registered_rotation, registered_translation)
                candidate_render, _ = gaussians.render(
                    render_cameras,
                    candidate_poses,
                    backgrounds=torch.ones(actual_views, 3, device="cuda"),
                )
                loss = torch.mean((candidate_render - target) ** 2)
                loss.backward()
                optimizer.step()
                value = float(torch.mean((candidate_render.detach() - target) ** 2).item())
                if value < best[0]:
                    best = (
                        value,
                        rotation_delta.detach().clone(),
                        translation_delta.detach().clone(),
                        log_scale.detach().clone(),
                    )

            registration_loss, best_rotation, best_translation, best_log_scale = best
            with torch.no_grad():
                global_rotation = rotation_matrix(best_rotation)
                registered_rotation = global_rotation[None] @ base_rotation
                registered_translation = (
                    torch.exp(best_log_scale) * (global_rotation[None] @ base_translation[..., None]).squeeze(-1)
                    + best_translation
                )
                c2w = torch.eye(4, device="cuda").repeat(actual_views, 1, 1)
                c2w[:, :3, :3] = registered_rotation
                c2w[:, :3, 3] = registered_translation
                current_rotation = registered_rotation.detach()
                current_translation = registered_translation.detach()

            # Match the official pose-free benchmark: starting from normalized GT poses,
            # refine every evaluation camera against its corresponding image. This happens
            # strictly after the pose-free Gaussian forward pass.
            if args.pose_refinement_steps > 0 and pose_lpips is None:
                import lpips
                pose_lpips = lpips.LPIPS(net="vgg").cuda().eval()
            rotation_update = torch.nn.Parameter(torch.zeros(actual_views, 3, device="cuda"))
            translation_update = torch.nn.Parameter(torch.zeros(actual_views, 3, device="cuda"))
            pose_optimizer = torch.optim.AdamW(
                [
                    {"params": [rotation_update], "lr": 0.005},
                    {"params": [translation_update], "lr": 0.005},
                ]
            )
            previous_loss = None
            patience = 0
            pose_steps = 0
            for pose_steps in range(1, args.pose_refinement_steps + 1):
                pose_optimizer.zero_grad()
                update_rotation = rotation_matrix(rotation_update)
                refined_rotation = current_rotation @ update_rotation
                refined_translation = current_translation + (
                    current_rotation @ translation_update[..., None]
                ).squeeze(-1)
                refined_poses = Pose.from_Rt(refined_rotation, refined_translation)
                refined_render, refined_info = gaussians.render(
                    render_cameras,
                    refined_poses,
                    backgrounds=torch.ones(actual_views, 3, device="cuda"),
                )
                mask = refined_info["alphas"]
                l1 = (torch.abs(refined_render - target) * mask).sum() / (mask.sum() * 3 + 1e-8)
                masked_render = refined_render * mask
                masked_target = target * mask
                perceptual = pose_lpips(masked_render * 2 - 1, masked_target * 2 - 1).mean()
                pose_loss = l1 + 0.05 * perceptual
                pose_loss.backward()
                pose_optimizer.step()
                with torch.no_grad():
                    applied_rotation = rotation_matrix(rotation_update)
                    current_translation = current_translation + (
                        current_rotation @ translation_update[..., None]
                    ).squeeze(-1)
                    current_rotation = current_rotation @ applied_rotation
                    rotation_update.zero_()
                    translation_update.zero_()
                value = float(pose_loss.item())
                if previous_loss is not None and abs(previous_loss - value) < 1e-5:
                    patience += 1
                    if patience >= 5:
                        break
                else:
                    patience = 0
                previous_loss = value
            with torch.no_grad():
                c2w = torch.eye(4, device="cuda").repeat(actual_views, 1, 1)
                c2w[:, :3, :3] = current_rotation
                c2w[:, :3, 3] = current_translation
                c2w = c2w.cpu().numpy()
            registration_scale = float(torch.exp(best_log_scale).item())
            registration_rotation = global_rotation.cpu().numpy().tolist()
            registration_translation = best_translation.cpu().numpy().tolist()
            poses = Pose.from_4x4mat(torch.from_numpy(c2w.astype(np.float32)))
            torch.cuda.synchronize()
            registration_seconds = time.perf_counter() - registration_start

        render_start = time.perf_counter()
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
            "posthoc_registration_seconds": registration_seconds,
            "posthoc_registration_scale": registration_scale,
            "posthoc_registration_convention": registration_convention if args.no_priors else None,
            "posthoc_registration_rotation": registration_rotation if args.no_priors else None,
            "posthoc_registration_translation": registration_translation if args.no_priors else None,
            "posthoc_registration_input_mse": registration_loss,
            "posthoc_pose_refinement_steps": pose_steps if args.no_priors else 0,
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
