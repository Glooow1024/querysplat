#!/usr/bin/env python3
"""Render QuerySplat 3DGS PLY files with the same ZipSplat turntable renderer."""

import argparse
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", type=int, required=True)
    parser.add_argument("--num-workers", type=int, required=True)
    parser.add_argument("--root", type=Path, action="append", required=True)
    parser.add_argument("--output-name", default="video_wide.mp4")
    parser.add_argument("--sweep-deg", type=float, default=120.0)
    args = parser.parse_args()

    import torch
    from plyfile import PlyData
    from zipsplat import Gaussians, viz

    jobs = []
    for root in args.root:
        jobs.extend(root.glob("[0-9][0-9]_*/[0-9]*views/output/gaussians.ply"))
    jobs = sorted(jobs)
    for ply_path in jobs[args.worker :: args.num_workers]:
        video_path = ply_path.parent / args.output_name
        if video_path.is_file():
            continue
        vertex = PlyData.read(ply_path)["vertex"].data
        names = vertex.dtype.names
        xyz = np.stack([vertex[k] for k in ("x", "y", "z")], -1)
        scales = np.exp(np.stack([vertex[k] for k in ("scale_0", "scale_1", "scale_2")], -1))
        quats = np.stack([vertex[f"rot_{i}"] for i in range(4)], -1)
        opacity = 1 / (1 + np.exp(-np.asarray(vertex["opacity"])))
        dc = np.stack([vertex[f"f_dc_{i}"] for i in range(3)], -1)[:, None, :]
        rest_names = sorted((n for n in names if n.startswith("f_rest_")), key=lambda n: int(n.rsplit("_", 1)[1]))
        if rest_names:
            rest = np.stack([vertex[n] for n in rest_names], -1)
            k_minus_1 = rest.shape[1] // 3
            rest = rest.reshape(len(rest), 3, k_minus_1).transpose(0, 2, 1)
            sh = np.concatenate([dc, rest], 1)
        else:
            sh = dc
        device = "cuda"
        g = Gaussians.from_parameters(
            means=torch.from_numpy(xyz).float().to(device),
            scales=torch.from_numpy(scales).float().to(device),
            quats=torch.from_numpy(quats).float().to(device),
            opacities=torch.from_numpy(opacity).float().to(device),
            sh_coeffs=torch.from_numpy(sh).float().to(device),
        )
        viz.turntable(
            g, video_path, render_size=384, num_frames=96, fps=24,
            sweep_deg=args.sweep_deg, elevation_deg=5.0, chunk=8,
        )
        print(f"DONE {video_path}", flush=True)


if __name__ == "__main__":
    main()
