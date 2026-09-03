#!/usr/bin/env python
# Copyright (c) 2026 Inspatio. SPDX-License-Identifier: Apache-2.0
"""Convert DL3DV-Evaluation (nerfstudio raw) into C3G's RE10K-style chunk format.

Mirrors C3G's src/scripts/convert_dl3dv.py so the output can be consumed by both
C3G (DatasetRE10k, which also loads dl3dv) and the QuerySplat eval script.

Outputs (under --out):
  test/000000.torch, 000001.torch, ...   # chunks of examples
  test/index.json                         # {scene_key: chunk_file}
And writes an evaluation index (--eval_index) with per-scene context/target.

Usage:
  python scripts/convert_dl3dv_eval.py \
    --root /data/datasets/3dvision/DL3DV-Evaluation/images \
    --out   /data/datasets/3dvision/DL3DV-Evaluation/c3g_eval \
    --eval_index /data/datasets/3dvision/DL3DV-Evaluation/eval_index_dl3dv.json \
    --num_target 10
"""
import argparse
import io
import json
import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image


def opengl_c2w_to_opencv_w2c(c2w: np.ndarray) -> np.ndarray:
    c2w = c2w.copy()
    c2w[2, :] *= -1
    c2w = c2w[np.array([1, 0, 2, 3]), :]
    c2w[0:3, 1:3] *= -1
    return np.linalg.inv(c2w)


def load_scene(scene_dir: Path):
    """Returns (cameras [N,18] float32, images list[raw bytes], frame_ids list[int])."""
    ns = scene_dir / "nerfstudio"
    meta = json.load(open(ns / "transforms.json"))
    w, h = meta["w"], meta["h"]
    fx = meta["fl_x"] / w
    fy = meta["fl_y"] / h
    cx = meta["cx"] / w
    cy = meta["cy"] / h
    intrinsic = np.array([fx, fy, cx, cy, 0.0, 0.0], dtype=np.float32)

    img8 = ns / "images_8"
    cameras, images, frame_ids = [], [], []
    for fr in meta["frames"]:
        fid = int(fr["file_path"].split("_")[-1].split(".")[0])
        p = img8 / f"frame_{fid:0>5}.png"
        # Skip missing / 0-byte / undecodable frames (a few source PNGs are corrupt),
        # and keep cameras/images/frame_ids in sync.
        if not p.exists() or p.stat().st_size == 0:
            continue
        with open(p, "rb") as f:
            raw = f.read()
        try:
            Image.open(io.BytesIO(raw)).convert("RGB")
        except Exception:
            continue
        c2w = np.array(fr["transform_matrix"], dtype=np.float32)
        w2c = opengl_c2w_to_opencv_w2c(c2w)[:3, :].flatten()
        cameras.append(np.concatenate([intrinsic, w2c]))
        images.append(torch.tensor(np.frombuffer(raw, dtype=np.uint8)))
        frame_ids.append(fid)
    cameras = torch.tensor(np.stack(cameras), dtype=torch.float32)
    return cameras, images, frame_ids


def make_eval_index(num_frames: int, num_target: int):
    """context=[0, F-1] (pair, expanded to N views by view_sampler); target=evenly spaced interior frames."""
    F = num_frames
    context = [0, F - 1]
    # target views offset from context endpoints, evenly spaced in the interior
    fracs = np.linspace(0.05, 0.95, num_target)
    target = [int(round(f * (F - 1))) for f in fracs]
    # dedupe & sort
    target = sorted(set(target))
    return {"context": context, "target": target, "overlap": None}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True, help="DL3DV-Evaluation/images dir")
    p.add_argument("--out", required=True, help="output dir (chunks written to <out>/test)")
    p.add_argument("--eval_index", required=True, help="output evaluation index json path")
    p.add_argument("--num_target", type=int, default=10)
    p.add_argument("--scenes_per_chunk", type=int, default=10)
    args = p.parse_args()

    root = Path(args.root)
    out_test = Path(args.out) / "test"
    out_test.mkdir(parents=True, exist_ok=True)

    scenes = sorted(d for d in root.iterdir() if d.is_dir())
    print(f"Found {len(scenes)} scenes")

    eval_index = {}
    chunk, chunk_idx, chunk_size = [], 0, 0
    TARGET_BYTES = int(1e8)

    def save_chunk():
        nonlocal chunk, chunk_idx, chunk_size
        if not chunk:
            return
        name = f"{chunk_idx:0>6}.torch"
        torch.save(chunk, out_test / name)
        print(f"  saved chunk {name} ({len(chunk)} scenes, {chunk_size/1e6:.1f} MB)")
        chunk_idx += 1
        chunk = []
        chunk_size = 0

    index = {}
    for sd in scenes:
        # scene dir is <hash>/<hash>
        inner = sd / sd.name
        if not (inner / "nerfstudio" / "transforms.json").exists():
            print(f"skip {sd.name}: no nerfstudio/transforms.json")
            continue
        try:
            cameras, images, frame_ids = load_scene(inner)
        except Exception as e:
            print(f"skip {sd.name}: {e}")
            continue
        key = sd.name
        example = {
            "key": key,
            "cameras": cameras,
            "images": images,
            "timestamps": torch.tensor(frame_ids, dtype=torch.int64),
        }
        chunk.append(example)
        index[key] = None  # filled after chunk save
        eval_index[key] = make_eval_index(len(frame_ids), args.num_target)
        chunk_size += sum(im.numel() for im in images)
        if chunk_size >= TARGET_BYTES or len(chunk) >= args.scenes_per_chunk:
            # assign chunk name to all scenes in this chunk
            cname = f"{chunk_idx:0>6}.torch"
            for ex in chunk:
                index[ex["key"]] = cname
            save_chunk()

    if chunk:
        cname = f"{chunk_idx:0>6}.torch"
        for ex in chunk:
            index[ex["key"]] = cname
        save_chunk()

    with open(out_test / "index.json", "w") as f:
        json.dump(index, f)
    with open(args.eval_index, "w") as f:
        json.dump(eval_index, f)
    print(f"Wrote {out_test/'index.json'} ({len(index)} scenes) and {args.eval_index}")


if __name__ == "__main__":
    main()
