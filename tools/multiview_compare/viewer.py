#!/usr/bin/env python3
"""Interactive multi-model trajectory and render viewer.

Run without arguments to choose an experiment folder with a native folder dialog:

    python tools/multiview_compare/viewer.py

Or pass a folder containing ``input/`` and ``output/`` directly:

    python tools/multiview_compare/viewer.py /root/multiview_compare/experiments
"""

from __future__ import annotations

import argparse
import json
import mimetypes
import os
import re
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse

import numpy as np


INDEX_RE = re.compile(r"view(\d+)")
FRAME_RE = re.compile(r"frame_\d+")
GT_TO_OPENCV = np.diag([1.0, -1.0, -1.0])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="View QuerySplat predicted cameras, reference cameras, inputs, and renders."
    )
    parser.add_argument("experiment", nargs="?", type=Path)
    parser.add_argument(
        "--transforms",
        type=Path,
        help="Optional Nerfstudio transforms.json. Auto-detected when omitted.",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=0, help="0 chooses a free port.")
    parser.add_argument("--no-browser", action="store_true")
    parser.add_argument(
        "--inspect",
        action="store_true",
        help="Validate files and print a summary without starting the viewer.",
    )
    return parser.parse_args()


def choose_folder() -> Path:
    try:
        import tkinter as tk
        from tkinter import filedialog
    except ImportError as exc:
        raise RuntimeError("Tkinter is unavailable; pass the experiment path on the command line.") from exc
    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    selected = filedialog.askdirectory(title="选择多视角重建实验目录")
    root.destroy()
    if not selected:
        raise SystemExit("No experiment folder selected.")
    return Path(selected)


def resolve_experiments(path: Path) -> tuple[dict[int, Path], int]:
    path = path.expanduser().resolve()
    candidates: list[Path] = [path]
    if path.name == "output":
        candidates.append(path.parent)
    direct = next(
        (p for p in candidates if (p / "output" / "predicted_input_cameras.json").is_file()),
        None,
    )
    group = direct.parent if direct else path
    if group.is_dir():
        candidates.extend(p for p in group.iterdir() if p.is_dir() and p.name.endswith("views"))
    valid: dict[int, Path] = {}
    for candidate in candidates:
        match = re.fullmatch(r"(\d+)views", candidate.name)
        if match and (candidate / "output" / "predicted_input_cameras.json").is_file():
            valid[int(match.group(1))] = candidate
    if not valid:
        raise FileNotFoundError(
            f"No predicted_input_cameras.json found below {path}. "
            "Choose a scene folder or a view folder such as 16views."
        )
    initial = int(direct.name.removesuffix("views")) if direct else max(valid)
    return dict(sorted(valid.items())), initial


def find_transforms(experiment: Path, explicit: Path | None) -> Path | None:
    if explicit:
        result = explicit.expanduser().resolve()
        if not result.is_file():
            raise FileNotFoundError(result)
        return result
    candidates = []
    for parent in [experiment, *list(experiment.parents)[:4]]:
        candidates.extend(
            [
                parent / "transforms.json",
                parent / "ground_truth" / "transforms.json",
                parent / "nerfstudio" / "transforms.json",
            ]
        )
    found = next((p for p in candidates if p.is_file()), None)
    if found:
        return found
    # The benchmark inputs are often symlinks into a Nerfstudio scene. Follow one
    # source image to find that scene's transforms without copying the dataset.
    for folder in (experiment / "input", experiment / "output" / "input_frames"):
        if folder.is_dir():
            for image in folder.iterdir():
                if image.is_file():
                    for parent in image.resolve().parents:
                        candidate = parent / "transforms.json"
                        if candidate.is_file():
                            return candidate
                    break
    return None


def frame_name(value: str) -> str | None:
    match = FRAME_RE.search(Path(value).name)
    return match.group(0) if match else None


def frame_number(value: str) -> int:
    name = frame_name(value)
    return int(name.rsplit("_", 1)[1]) if name else 10**12


def sim3(source: np.ndarray, target: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    """Return scale, rotation, translation mapping source points to target points."""
    if len(source) < 3:
        raise ValueError("At least three matched cameras are required for Sim(3) alignment.")
    source_mean, target_mean = source.mean(0), target.mean(0)
    source_centered, target_centered = source - source_mean, target - target_mean
    covariance = target_centered.T @ source_centered / len(source)
    u, singular, vt = np.linalg.svd(covariance)
    sign = np.eye(3)
    sign[-1, -1] = np.sign(np.linalg.det(u @ vt))
    rotation = u @ sign @ vt
    variance = np.sum(source_centered * source_centered) / len(source)
    scale = float(np.trace(np.diag(singular) @ sign) / variance)
    translation = target_mean - scale * rotation @ source_mean
    return scale, rotation, translation


def indexed_files(folder: Path) -> dict[int, Path]:
    result: dict[int, Path] = {}
    if not folder.is_dir():
        return result
    for path in folder.iterdir():
        match = INDEX_RE.search(path.name)
        if path.is_file() and match:
            result[int(match.group(1))] = path
    return result


def ply_vertex_count(path: Path) -> int | None:
    if not path.is_file():
        return None
    with path.open("rb") as handle:
        header = handle.read(8192).decode("ascii", errors="ignore")
    match = re.search(r"element vertex (\d+)", header)
    return int(match.group(1)) if match else None


def load_experiment(experiment: Path, transforms_path: Path | None) -> tuple[dict, dict[str, Path]]:
    output = experiment / "output"
    prediction = json.loads((output / "predicted_input_cameras.json").read_text(encoding="utf-8"))
    predicted_frames = prediction.get("frames", [])
    if not predicted_frames:
        raise ValueError("predicted_input_cameras.json contains no frames.")

    inputs = indexed_files(output / "input_frames")
    renders = indexed_files(output / "rendered")
    full_renders = indexed_files(output / "rendered_full_gs")
    filtered_renders = indexed_files(output / "rendered_filtered_gs")
    gt_aligned_renders = indexed_files(output / "rendered_gt_aligned")
    media: dict[str, Path] = {}
    for index, path in inputs.items():
        media[f"input/{index}"] = path
    for index, path in renders.items():
        media[f"render/{index}"] = path
    for index, path in full_renders.items():
        media[f"render_full/{index}"] = path
    for index, path in filtered_renders.items():
        media[f"render_filtered/{index}"] = path
    for index, path in gt_aligned_renders.items():
        media[f"render_gt/{index}"] = path
    pointcloud = output / "pointcloud.ply"
    if pointcloud.is_file():
        media["pointcloud"] = pointcloud
    full_manifest = output / "full_trajectory_manifest.json"
    full_gs_manifest = output / "full_trajectory_full_gs_manifest.json"
    full_is_gt = False
    if full_manifest.is_file():
        try:
            full_is_gt = json.loads(full_manifest.read_text()).get("trajectory_source") == "aligned_full_ground_truth"
        except (OSError, json.JSONDecodeError):
            pass
    full_gs_is_gt = False
    if full_gs_manifest.is_file():
        try:
            full_gs_is_gt = json.loads(full_gs_manifest.read_text()).get("trajectory_source") == "aligned_full_ground_truth"
        except (OSError, json.JSONDecodeError):
            pass
    videos = {
        key: path
        for key, path in (
            ("predicted_full_gs", output / "video_predicted_trajectory_full_gs.mp4"),
            ("predicted", output / "video_predicted_trajectory.mp4"),
            ("ground_truth_full_gs", output / "video_full_trajectory_full_gs.mp4" if full_gs_is_gt else Path("/__missing__")),
            ("ground_truth", output / "video_full_trajectory.mp4" if full_is_gt else Path("/__missing__")),
            ("wide", output / "video_wide.mp4"),
            ("wiggle", output / "video.mp4"),
            ("turntable", output / "turntable.mp4"),
        )
        if path.is_file()
    }
    for key, path in videos.items():
        media[f"video/{key}"] = path

    result: dict = {
        "experiment": str(experiment),
        "view_count": len(predicted_frames),
        "has_ground_truth": False,
        "reference_path": str(transforms_path) if transforms_path else None,
        "full_gt": [],
        "frames": [],
        "metrics": {},
        "reference_warning": None,
        "video_options": list(videos),
        "render_options": (["full_gs"] if full_renders else []) + (["filtered_gs"] if filtered_renders else []) + (["native"] if renders else []) + (["gt_aligned"] if gt_aligned_renders else []),
        "reconstruction": {
            "gaussian_count": ply_vertex_count(output / "gaussians.ply") or ply_vertex_count(output / "gaussians_opacity0.05.ply"),
            "gaussian_count_full": ply_vertex_count(output / "gaussians_opacity0.ply") or ply_vertex_count(output / "gaussians.ply"),
            "gaussian_filter": "opacity >= 0.05",
            "pointcloud_count": ply_vertex_count(output / "pointcloud.ply"),
        },
    }
    predicted_by_name = {f["name"]: f for f in predicted_frames}

    def append_unaligned_predictions() -> None:
        for f in predicted_frames:
            raw_c2w = np.asarray(f["c2w"], dtype=float)
            result["frames"].append(
                {
                    "index": int(f["index"]),
                    "name": f["name"],
                    "pred": np.asarray(f["c2w"], dtype=float)[:3, 3].round(6).tolist(),
                    "pred_dir": np.asarray(f["c2w"], dtype=float)[:3, 2].round(6).tolist(),
                    "scene_c2w": raw_c2w[:3, :4].round(6).tolist(),
                    "has_input": int(f["index"]) in inputs,
                    "has_render": int(f["index"]) in renders,
                    "has_render_full": int(f["index"]) in full_renders,
                    "has_render_filtered": int(f["index"]) in filtered_renders,
                    "has_render_gt": int(f["index"]) in gt_aligned_renders,
                }
            )

    if transforms_path is None:
        append_unaligned_predictions()
        return result, media

    reference = json.loads(transforms_path.read_text(encoding="utf-8"))
    reference_frames = reference.get("frames", [])
    ordered_reference_frames = sorted(
        reference_frames, key=lambda f: frame_number(f.get("file_path", ""))
    )
    gt_by_name = {
        name: np.asarray(f["transform_matrix"], dtype=float)
        for f in reference_frames
        if (name := frame_name(f.get("file_path", "")))
    }
    matched_names = sorted(set(predicted_by_name) & set(gt_by_name))
    if len(matched_names) < 3:
        result["reference_warning"] = (
            f"Only {len(matched_names)} predicted frames match the reference; "
            "showing the unaligned prediction trajectory only."
        )
        append_unaligned_predictions()
        return result, media

    source = np.stack(
        [np.asarray(predicted_by_name[name]["c2w"], dtype=float)[:3, 3] for name in matched_names]
    )
    target = np.stack([gt_by_name[name][:3, 3] for name in matched_names])
    scale, rotation, translation = sim3(source, target)
    aligned = (scale * (rotation @ source.T)).T + translation

    result["has_ground_truth"] = True
    result["full_gt"] = [
        {
            "name": frame_name(f.get("file_path", "")) or f"frame_{i:05d}",
            "p": np.asarray(f["transform_matrix"], dtype=float)[:3, 3].round(6).tolist(),
        }
        for i, f in enumerate(ordered_reference_frames)
    ]
    position_errors, rotation_errors = [], []
    aligned_by_name: dict[str, tuple[np.ndarray, np.ndarray, float, float]] = {}
    for i, name in enumerate(matched_names):
        gt_c2w = gt_by_name[name]
        pred_c2w = np.asarray(predicted_by_name[name]["c2w"], dtype=float)
        gt_rotation = gt_c2w[:3, :3] @ GT_TO_OPENCV
        pred_rotation = rotation @ pred_c2w[:3, :3]
        relative = gt_rotation.T @ pred_rotation
        angle = float(np.degrees(np.arccos(np.clip((np.trace(relative) - 1.0) / 2.0, -1.0, 1.0))))
        error = float(np.linalg.norm(aligned[i] - target[i]))
        position_errors.append(error)
        rotation_errors.append(angle)
        aligned_by_name[name] = (aligned[i], pred_rotation[:, 2], error, angle)

    for f in predicted_frames:
        index, name = int(f["index"]), f["name"]
        raw = np.asarray(f["c2w"], dtype=float)
        item = {
            "index": index,
            "name": name,
            "has_input": index in inputs,
            "has_render": index in renders,
            "has_render_full": index in full_renders,
            "has_render_filtered": index in filtered_renders,
            "has_render_gt": index in gt_aligned_renders,
            "matched": name in aligned_by_name,
            "scene_c2w": raw[:3, :4].round(6).tolist(),
        }
        if name in aligned_by_name:
            aligned_position, aligned_direction, error, angle = aligned_by_name[name]
            gt_c2w = gt_by_name[name]
            gt_rotation_opencv = gt_c2w[:3, :3] @ GT_TO_OPENCV
            scene_gt_position = rotation.T @ ((gt_c2w[:3, 3] - translation) / scale)
            scene_gt_rotation = rotation.T @ gt_rotation_opencv
            scene_gt_c2w = np.concatenate(
                [scene_gt_rotation, scene_gt_position[:, None]], axis=1
            )
            item.update(
                {
                    "gt": gt_c2w[:3, 3].round(6).tolist(),
                    "pred": aligned_position.round(6).tolist(),
                    "gt_dir": (gt_c2w[:3, :3] @ GT_TO_OPENCV)[:, 2].round(6).tolist(),
                    "pred_dir": aligned_direction.round(6).tolist(),
                    "scene_gt_c2w": scene_gt_c2w.round(6).tolist(),
                    "position_error": round(error, 6),
                    "rotation_error_deg": round(angle, 3),
                }
            )
        else:
            item.update(
                {
                    "pred": (scale * rotation @ raw[:3, 3] + translation).round(6).tolist(),
                    "pred_dir": (rotation @ raw[:3, :3])[:, 2].round(6).tolist(),
                }
            )
        result["frames"].append(item)

    result["metrics"] = {
        "reference_frames": len(reference_frames),
        "matched_frames": len(matched_names),
        "scale_pred_to_gt": round(scale, 6),
        "ate_rmse": round(float(np.sqrt(np.mean(np.square(position_errors)))), 6),
        "mean_position_error": round(float(np.mean(position_errors)), 6),
        "mean_rotation_error_deg": round(float(np.mean(rotation_errors)), 3),
        "max_position_error": round(float(np.max(position_errors)), 6),
    }
    return result, media


HTML = r"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>多视角重建对比检查器</title>
<style>
:root{color-scheme:light dark;--bg:#f5f6f8;--panel:#fff;--fg:#172033;--muted:#687386;--line:#d9dee8;--gt:#2878d0;--pred:#e47728;--track:#8893a3;--bad:#c83e4d}
@media(prefers-color-scheme:dark){:root{--bg:#11151d;--panel:#1a202b;--fg:#edf2fa;--muted:#9ca8b9;--line:#364052;--gt:#62aaf3;--pred:#f5a052;--track:#778397;--bad:#ff6f7d}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font:14px/1.45 system-ui,-apple-system,"Segoe UI",sans-serif}
header{padding:12px 18px;border-bottom:1px solid var(--line);background:var(--panel);display:flex;align-items:center;gap:16px;flex-wrap:wrap}h1{font-size:17px;margin:0;font-weight:600}.meta{color:var(--muted);font-size:12px;overflow-wrap:anywhere}
.layout{display:grid;grid-template-columns:minmax(440px,1fr) minmax(330px,440px);gap:12px;padding:12px;height:calc(100vh - 65px);min-height:600px}.panel{background:var(--panel);border:1px solid var(--line);border-radius:10px;overflow:hidden}.left{display:flex;flex-direction:column}.toolbar{padding:9px 12px;border-bottom:1px solid var(--line);display:flex;gap:8px;align-items:center;flex-wrap:wrap}button{font:inherit;color:var(--fg);background:transparent;border:1px solid var(--line);border-radius:7px;padding:6px 9px;cursor:pointer}button[aria-pressed=true]{background:color-mix(in srgb,var(--gt) 13%,transparent);border-color:var(--gt)}button:hover{border-color:var(--muted)}.sw{display:inline-block;width:9px;height:9px;margin-right:5px}.sw.gt{background:var(--gt);border-radius:50%}.sw.pred{background:var(--pred);transform:rotate(45deg)}.sw.track{background:var(--track)}
#plot,#sceneCanvas{width:100%;height:100%;min-height:400px;touch-action:none;cursor:grab}#plot:active,#sceneCanvas:active{cursor:grabbing}#sceneCanvas{display:none;background:var(--bg)}.frame{fill:transparent;stroke:var(--line)}.grid{stroke:var(--line);opacity:.55}.axis{stroke:var(--muted)}.axisText{fill:var(--fg);font-size:12px}.full{fill:none;stroke:var(--track);stroke-width:1.4;opacity:.75}.gtPath{fill:none;stroke:var(--gt);stroke-width:2}.predPath{fill:none;stroke:var(--pred);stroke-width:2;stroke-dasharray:5 4}.connector{stroke:var(--line)}.gtPoint{fill:var(--gt);stroke:var(--panel);stroke-width:1.5;cursor:pointer}.predPoint{fill:var(--pred);stroke:var(--panel);stroke-width:1.5;cursor:pointer}.selected{stroke:var(--fg);stroke-width:3}.gtDir{stroke:var(--gt);stroke-width:1.5}.predDir{stroke:var(--pred);stroke-width:1.5;stroke-dasharray:3 2}
.right{display:flex;flex-direction:column;min-height:0}.details{padding:10px 12px;border-bottom:1px solid var(--line)}.view-nav{display:grid;grid-template-columns:auto minmax(120px,1fr) auto;gap:7px;margin-bottom:9px}.frame-title{font-size:16px;font-weight:600}.errors{margin-top:4px;color:var(--muted)}.mode-info{padding:8px 12px;border-bottom:1px solid var(--line);display:flex;gap:14px;align-items:center;flex-wrap:wrap;color:var(--muted);font-size:12px}.mode-info b{color:var(--fg);font-weight:600}.instructions{flex-basis:100%;display:grid;grid-template-columns:repeat(4,minmax(120px,1fr));gap:6px}.gesture{padding:6px 8px;background:color-mix(in srgb,var(--gt) 9%,transparent);border-left:3px solid var(--gt);border-radius:5px;color:var(--muted)}.gesture b{display:block;color:var(--fg)}.camera-key{display:inline-block;width:10px;height:10px;margin:0 4px 0 8px}.camera-key.gt{background:var(--gt)}.camera-key.pred{background:var(--pred)}#sceneInfo{display:none}.images{padding:10px;overflow:auto;display:grid;gap:12px}.image-block h2{font-size:13px;margin:0 0 5px;color:var(--muted);font-weight:500}.image-block img,.image-block video{display:block;width:100%;aspect-ratio:1;background:var(--bg);object-fit:contain;border:1px solid var(--line);border-radius:7px}.missing{width:100%;aspect-ratio:1;display:grid;place-items:center;border:1px dashed var(--line);border-radius:7px;color:var(--muted)}.stats{margin-left:auto;display:flex;gap:12px;color:var(--muted);font-size:12px}.stats b{color:var(--fg);font-weight:600}.warning{color:var(--bad)}
select{font:inherit;color:var(--fg);background:var(--panel);border:1px solid var(--line);border-radius:7px;padding:6px 28px 6px 8px}
@media(max-width:850px){.layout{grid-template-columns:1fr;height:auto}.left{height:700px}.right{max-height:none}.stats{margin-left:0}.images{grid-template-columns:1fr 1fr}.layout{min-height:0}.instructions{grid-template-columns:repeat(2,minmax(120px,1fr))}}
@media(max-width:560px){.layout{padding:6px}.left{height:520px}.images{grid-template-columns:1fr}}
</style>
</head>
<body>
<header><div><h1>QuerySplat / ZipSplat 对比检查器</h1><div id="experiment" class="meta"></div></div><div id="stats" class="stats"></div></header>
<main class="layout">
 <section class="panel left">
  <div class="toolbar">
   <label for="datasetSelect">数据组</label><select id="datasetSelect"></select>
   <label for="sceneSelect">场景</label><select id="sceneSelect"></select>
   <label for="viewSelect">视角数</label><select id="viewSelect"></select>
   <label for="methodSelect">方法</label><select id="methodSelect"></select>
   <button id="trajectoryMode" aria-pressed="true">轨迹</button><button id="sceneMode" aria-pressed="false">三维场</button>
   <button data-layer="full" aria-pressed="true"><span class="sw track"></span>完整真值</button>
   <button data-layer="gt" aria-pressed="true"><span class="sw gt"></span>输入帧真值</button>
   <button data-layer="pred" aria-pressed="true"><span class="sw pred"></span>预测轨迹</button>
   <button data-layer="dirs" aria-pressed="true">朝向</button>
   <button id="reset">重置视角</button>
  </div>
  <div id="trajectoryInfo" class="mode-info"><span>平均位置误差 <b id="meanPos">—</b></span><span>ATE RMSE <b id="ateValue">—</b></span><span>平均旋转误差 <b id="meanRot">—</b></span><span>当前位置误差 <b id="currentPos">—</b></span><span>当前旋转误差 <b id="currentRot">—</b></span></div>
  <div id="sceneInfo" class="mode-info"><div class="instructions"><div class="gesture"><b>左键拖动</b>旋转场景</div><div class="gesture"><b>Shift+左键 / 右键</b>平移场景</div><div class="gesture"><b>鼠标滚轮</b>朝光标位置缩放</div><div class="gesture"><b>重置视角</b>恢复旋转、平移与缩放</div></div><span>相机显示 <select id="cameraDisplay"><option value="current" selected>当前相机</option><option value="all">所有相机</option><option value="none">隐藏相机</option></select><i class="camera-key gt"></i>真值（实线）<i class="camera-key pred"></i>预测（虚线）</span><span>点数上限 <select id="pointBudget"><option value="20000">2万（最快）</option><option value="50000" selected>5万（默认）</option><option value="100000">10万</option><option value="0">全部点（较慢）</option></select></span><span>全量 GS <b id="gaussianFullCount">—</b></span><span>筛选 GS（opacity ≥ 0.05） <b id="gaussianCount">—</b></span><span>彩色点数 <b id="pointCount">—</b></span><span>浏览器当前显示 <b id="displayedPointCount">—</b></span><span>缩放 <b id="sceneZoomValue">1.00×</b></span></div>
  <svg id="plot" role="img" aria-label="可拖动的真值与预测相机轨迹"></svg>
  <canvas id="sceneCanvas" aria-label="可拖动旋转的彩色三维点云场"></canvas>
 </section>
 <aside class="panel right">
  <div class="details"><div class="view-nav"><button id="prevFrame">上一视角</button><select id="frameSelect" aria-label="选择相机视角"></select><button id="nextFrame">下一视角</button></div><div id="frameTitle" class="frame-title">选择一个相机节点</div><div id="errors" class="errors">点击轨迹节点或使用上方控件切换图像</div></div>
  <div class="images">
   <div class="image-block"><h2>真实输入（预处理后）</h2><div id="inputSlot"></div></div>
   <div class="image-block"><h2>当前算法渲染结果　<select id="renderSelect"><option value="predicted">预测位姿</option><option value="gt_aligned">Sim(3) 对齐真实位姿</option></select></h2><div id="renderSlot"></div></div>
   <div class="image-block"><h2>当前算法渲染视频　<select id="videoSelect"></select></h2><div id="videoSlot"></div></div>
  </div>
 </aside>
</main>
<script>
const NS='http://www.w3.org/2000/svg',plot=document.getElementById('plot');
let data,yaw=-.55,pitch=-.35,zoom=1,drag=false,last,selected=null;
const sceneCanvas=document.getElementById('sceneCanvas'),sceneCtx=sceneCanvas.getContext('2d');
let displayMode='trajectory',sceneYaw=-.55,scenePitch=-.3,sceneZoom=1,scenePanX=0,scenePanY=0,sceneDrag=false,scenePanDrag=false,sceneLast=null,sceneCloud=null;
const sceneCache=new Map();
const sceneBufferCache=new Map();
const visible={full:true,gt:true,pred:true,dirs:true};
const E=(tag,a={},text='')=>{const n=document.createElementNS(NS,tag);Object.entries(a).forEach(([k,v])=>n.setAttribute(k,v));if(text)n.textContent=text;return n};
const rotate=p=>{let[x,y,z]=p,cy=Math.cos(yaw),sy=Math.sin(yaw),cx=Math.cos(pitch),sx=Math.sin(pitch),xx=cy*x+sy*z,zz=-sy*x+cy*z;return[xx,cx*y-sx*zz,sx*y+cx*zz]};
function selectFrame(frame){selected=frame.index;const fs=document.getElementById('frameSelect');fs.value=String(frame.index);const pos=data.frames.findIndex(f=>f.index===frame.index);document.getElementById('prevFrame').disabled=pos<=0;document.getElementById('nextFrame').disabled=pos>=data.frames.length-1;document.getElementById('frameTitle').textContent=`${frame.name} · view ${frame.index}`;const e=document.getElementById('errors');e.textContent=frame.matched?`位置误差 ${frame.position_error.toFixed(3)} · 旋转误差 ${frame.rotation_error_deg.toFixed(2)}°`:'该帧没有对应的参考位姿';document.getElementById('currentPos').textContent=frame.matched?frame.position_error.toFixed(4):'—';document.getElementById('currentRot').textContent=frame.matched?frame.rotation_error_deg.toFixed(2)+'°':'—';setImage('inputSlot',frame.has_input?`/media/${data.selected_id}/input/${frame.index}`:null,'没有输入图片');const mode=document.getElementById('renderSelect').value,map={full_gs:['has_render_full','render_full'],filtered_gs:['has_render_filtered','render_filtered'],native:['has_render','render'],gt_aligned:['has_render_gt','render_gt']},choice=map[mode]||map.native,hasRender=frame[choice[0]],kind=choice[1];setImage('renderSlot',hasRender?`/media/${data.selected_id}/${kind}/${frame.index}`:null,mode==='gt_aligned'?'该视角没有真实位姿或尚未渲染':'没有渲染图片');if(displayMode==='scene')renderScene();else draw()}
function populateFrameSelect(){const fs=document.getElementById('frameSelect');fs.replaceChildren();data.frames.forEach(f=>{const o=document.createElement('option');o.value=f.index;o.textContent=`view ${f.index} · ${f.name}`;fs.append(o)});fs.onchange=()=>selectFrame(data.frames.find(f=>f.index===Number(fs.value)))}
function stepFrame(delta){const pos=Math.max(0,data.frames.findIndex(f=>f.index===selected)),next=Math.max(0,Math.min(data.frames.length-1,pos+delta));selectFrame(data.frames[next])}
function setImage(id,url,message){const s=document.getElementById(id);s.replaceChildren();if(url){const im=document.createElement('img');im.src=url+'?v='+Date.now();im.alt=id==='inputSlot'?'真实输入图片':'当前算法渲染结果';s.append(im)}else{const d=document.createElement('div');d.className='missing';d.textContent=message;s.append(d)}}
function setRenderOptions(){const sel=document.getElementById('renderSelect'),labels={full_gs:'预测位姿 · 全量 GS（opacity ≥ 0）',filtered_gs:'预测位姿 · 筛选 GS（opacity ≥ 0.05）',native:'预测位姿 · 仓库原生渲染（模型默认）',gt_aligned:'Sim(3) 对齐真实位姿'};sel.replaceChildren();(data.render_options||[]).forEach(k=>{const o=document.createElement('option');o.value=k;o.textContent=labels[k]||k;sel.append(o)});sel.value=(data.render_options||[])[0]||'native'}
function setVideo(){const s=document.getElementById('videoSlot'),sel=document.getElementById('videoSelect'),labels={predicted_full_gs:'预测位姿全程插值 · 全量 GS（opacity ≥ 0）',predicted:'预测位姿全程插值 · 筛选 GS（opacity ≥ 0.05）',ground_truth_full_gs:'真实位姿全程轨迹 · 全量 GS（opacity ≥ 0）',ground_truth:'真实位姿全程轨迹 · 筛选 GS（opacity ≥ 0.05）',wide:'120°轨道',wiggle:'小范围摆动',turntable:'转台'};s.replaceChildren();sel.replaceChildren();(data.video_options||[]).forEach(k=>{const o=document.createElement('option');o.value=k;o.textContent=labels[k]||k;sel.append(o)});if((data.video_options||[]).length){sel.value=data.video_options.includes('predicted_full_gs')?'predicted_full_gs':(data.video_options.includes('predicted')?'predicted':data.video_options[0]);const show=()=>{s.replaceChildren();const v=document.createElement('video');v.controls=true;v.loop=true;v.preload='metadata';v.src=`/media/${data.selected_id}/video/${sel.value}?v=${Date.now()}`;s.append(v)};sel.onchange=show;show()}else{const d=document.createElement('div');d.className='missing';d.textContent='没有渲染视频';s.append(d)}}
function parsePly(buffer,maxPoints){const bytes=new Uint8Array(buffer),limit=Math.min(bytes.length,8192),text=new TextDecoder().decode(bytes.subarray(0,limit)),marker='end_header',at=text.indexOf(marker);if(at<0)throw new Error('无效的 PLY 文件');let offset=at+marker.length;while(offset<bytes.length&&(bytes[offset]===10||bytes[offset]===13))offset++;const count=Number((text.match(/element vertex (\d+)/)||[])[1]);if(!count)throw new Error('PLY 中没有顶点');const stride=15,step=maxPoints>0?Math.max(1,Math.ceil(count/maxPoints)):1,n=Math.ceil(count/step),pos=new Float32Array(n*3),col=new Uint8Array(n*3),dv=new DataView(buffer);let j=0,min=[Infinity,Infinity,Infinity],max=[-Infinity,-Infinity,-Infinity];for(let i=0;i<count;i+=step){const o=offset+i*stride,x=dv.getFloat32(o,true),y=dv.getFloat32(o+4,true),z=dv.getFloat32(o+8,true);pos[j*3]=x;pos[j*3+1]=y;pos[j*3+2]=z;col[j*3]=dv.getUint8(o+12);col[j*3+1]=dv.getUint8(o+13);col[j*3+2]=dv.getUint8(o+14);min[0]=Math.min(min[0],x);min[1]=Math.min(min[1],y);min[2]=Math.min(min[2],z);max[0]=Math.max(max[0],x);max[1]=Math.max(max[1],y);max[2]=Math.max(max[2],z);j++}const center=min.map((v,k)=>(v+max[k])/2),extent=Math.max(...max.map((v,k)=>v-min[k]))||1;for(let i=0;i<j;i++)for(let k=0;k<3;k++)pos[i*3+k]=(pos[i*3+k]-center[k])/extent;return{pos,col,count:j,sourceCount:count,center,extent}}
async function loadScene(){if(!data)return;const sourceKey=data.selected_id,budget=Number(document.getElementById('pointBudget').value),cacheKey=sourceKey+'/'+budget;if(sceneCache.has(cacheKey)){sceneCloud=sceneCache.get(cacheKey);document.getElementById('displayedPointCount').textContent=sceneCloud.count.toLocaleString();renderScene();return}sceneCloud=null;document.getElementById('displayedPointCount').textContent='加载中…';const errors=document.getElementById('errors');errors.textContent=sceneBufferCache.has(sourceKey)?'正在重新采样浏览器内存中的点云…':'正在从服务器加载彩色三维点云…';try{let buffer=sceneBufferCache.get(sourceKey);if(!buffer){const r=await fetch(`/media/${data.selected_id}/pointcloud`);if(!r.ok)throw new Error('该实验没有 pointcloud.ply');buffer=await r.arrayBuffer();sceneBufferCache.set(sourceKey,buffer)}const cloud=parsePly(buffer,budget);sceneCache.set(cacheKey,cloud);sceneCloud=cloud;document.getElementById('displayedPointCount').textContent=cloud.count.toLocaleString();errors.textContent=`三维场：${cloud.count.toLocaleString()} / ${cloud.sourceCount.toLocaleString()} 点`;renderScene()}catch(e){document.getElementById('displayedPointCount').textContent='—';errors.textContent=e.message;renderScene()}}
function sceneProject(p,W,H,scale){const x=p[0],y=-p[1],z=p[2],cy=Math.cos(sceneYaw),sy=Math.sin(sceneYaw),cx=Math.cos(scenePitch),sx=Math.sin(scenePitch),xx=cy*x+sy*z,zz=-sy*x+cy*z,yy=cx*y-sx*zz;return[W/2+scenePanX+xx*scale,H/2+scenePanY-yy*scale]}
function drawFrustum(m,color,isCurrent,size,dashed,W,H,scale,dpr){if(!m)return;const c=[0,1,2].map(k=>(m[k][3]-sceneCloud.center[k])/sceneCloud.extent),right=[m[0][0],m[1][0],m[2][0]],down=[m[0][1],m[1][1],m[2][1]],forward=[m[0][2],m[1][2],m[2][2]],corner=(sx,sy)=>c.map((v,k)=>v+forward[k]*size+right[k]*size*.55*sx+down[k]*size*.4*sy),corners=[corner(-1,-1),corner(1,-1),corner(1,1),corner(-1,1)],pc=sceneProject(c,W,H,scale),pts=corners.map(p=>sceneProject(p,W,H,scale));sceneCtx.strokeStyle=color;sceneCtx.fillStyle=color;sceneCtx.lineWidth=(isCurrent?2.6:1.15)*dpr;sceneCtx.globalAlpha=isCurrent?1:.72;sceneCtx.setLineDash(dashed?[5*dpr,3*dpr]:[]);sceneCtx.beginPath();pts.forEach(p=>{sceneCtx.moveTo(pc[0],pc[1]);sceneCtx.lineTo(p[0],p[1])});sceneCtx.moveTo(pts[0][0],pts[0][1]);for(let i=1;i<4;i++)sceneCtx.lineTo(pts[i][0],pts[i][1]);sceneCtx.closePath();sceneCtx.stroke();sceneCtx.setLineDash([]);sceneCtx.beginPath();sceneCtx.arc(pc[0],pc[1],(isCurrent?3.7:2.1)*dpr,0,Math.PI*2);sceneCtx.fill()}
function drawSceneCameras(W,H,scale){const mode=document.getElementById('cameraDisplay').value;if(mode==='none'||!data||!sceneCloud)return;const frames=mode==='current'?data.frames.filter(f=>f.index===selected):data.frames,style=getComputedStyle(document.documentElement),gtColor=style.getPropertyValue('--gt').trim(),predColor=style.getPropertyValue('--pred').trim(),dpr=Math.min(devicePixelRatio||1,1.5),frustumSize=.057;sceneCtx.lineJoin='round';frames.forEach(f=>{const current=f.index===selected;drawFrustum(f.scene_gt_c2w,gtColor,current,frustumSize,false,W,H,scale,dpr);drawFrustum(f.scene_c2w,predColor,current,frustumSize,true,W,H,scale,dpr)});sceneCtx.globalAlpha=1}
function renderScene(){if(displayMode!=='scene')return;document.getElementById('sceneZoomValue').textContent=sceneZoom.toFixed(2)+'×';const dpr=Math.min(devicePixelRatio||1,1.5),w=Math.max(320,sceneCanvas.clientWidth),h=Math.max(360,sceneCanvas.clientHeight);if(sceneCanvas.width!==Math.round(w*dpr)||sceneCanvas.height!==Math.round(h*dpr)){sceneCanvas.width=Math.round(w*dpr);sceneCanvas.height=Math.round(h*dpr)}const W=sceneCanvas.width,H=sceneCanvas.height;sceneCtx.clearRect(0,0,W,H);if(!sceneCloud)return;const image=sceneCtx.createImageData(W,H),pix=image.data,zbuf=new Float32Array(W*H);zbuf.fill(-Infinity);const p=sceneCloud.pos,c=sceneCloud.col,cy=Math.cos(sceneYaw),sy=Math.sin(sceneYaw),cx=Math.cos(scenePitch),sx=Math.sin(scenePitch),scale=Math.min(W,H)*.82*sceneZoom;for(let i=0;i<sceneCloud.count;i++){const x=p[i*3],y=-p[i*3+1],z=p[i*3+2],xx=cy*x+sy*z,zz=-sy*x+cy*z,yy=cx*y-sx*zz,depth=sx*y+cx*zz,px=Math.round(W/2+scenePanX+xx*scale),py=Math.round(H/2+scenePanY-yy*scale);if(px<1||px>=W-1||py<1||py>=H-1)continue;const zi=py*W+px;if(depth<=zbuf[zi])continue;for(let oy=0;oy<2;oy++)for(let ox=0;ox<2;ox++){const q=(py+oy)*W+px+ox,k=q*4;zbuf[q]=depth;pix[k]=c[i*3];pix[k+1]=c[i*3+1];pix[k+2]=c[i*3+2];pix[k+3]=255}}sceneCtx.putImageData(image,0,0);drawSceneCameras(W,H,scale)}
function setDisplayMode(mode){displayMode=mode;const isScene=mode==='scene';plot.style.display=isScene?'none':'block';sceneCanvas.style.display=isScene?'block':'none';document.getElementById('trajectoryInfo').style.display=isScene?'none':'flex';document.getElementById('sceneInfo').style.display=isScene?'flex':'none';document.getElementById('trajectoryMode').setAttribute('aria-pressed',String(!isScene));document.getElementById('sceneMode').setAttribute('aria-pressed',String(isScene));if(isScene)loadScene();else draw()}
function draw(){if(!data)return;const w=Math.max(320,plot.clientWidth),h=Math.max(360,plot.clientHeight);plot.setAttribute('viewBox',`0 0 ${w} ${h}`);plot.replaceChildren();const pad=28,cx=w/2,cy=h/2;plot.append(E('rect',{x:pad,y:8,width:w-pad-8,height:h-20,class:'frame'}));for(let i=1;i<4;i++){let yy=8+i*(h-28)/4;plot.append(E('line',{x1:pad,y1:yy,x2:w-8,y2:yy,class:'grid'}))}
 const matched=data.frames.filter(f=>f.matched),all=[...data.full_gt.map(x=>x.p),...data.frames.map(x=>x.pred)];if(!all.length)return;const center=[0,1,2].map(k=>all.reduce((s,p)=>s+p[k],0)/all.length),rp=all.map(p=>rotate(p.map((v,k)=>v-center[k]))),ex=Math.max(...rp.map(p=>Math.abs(p[0])),1e-6),ey=Math.max(...rp.map(p=>Math.abs(p[1])),1e-6),scale=Math.min((w-pad-35)/(2*ex),(h-55)/(2*ey))*zoom,project=p=>{const q=rotate(p.map((v,k)=>v-center[k]));return[cx+q[0]*scale,cy-q[1]*scale,q[2]]},path=pts=>pts.map((p,i)=>{const q=project(p);return(i?'L':'M')+q[0].toFixed(1)+','+q[1].toFixed(1)}).join(' ');
 const o=project(center),extent=Math.max(...all.flatMap(p=>p.map((v,k)=>Math.abs(v-center[k]))));[['X',[extent*.15,0,0]],['Y',[0,extent*.15,0]],['Z',[0,0,extent*.15]]].forEach(([l,d])=>{const q=project(center.map((v,k)=>v+d[k]));plot.append(E('line',{x1:o[0],y1:o[1],x2:q[0],y2:q[1],class:'axis'}));plot.append(E('text',{x:q[0]+3,y:q[1]-3,class:'axisText'},l))});
 if(visible.full&&data.full_gt.length)plot.append(E('path',{d:path(data.full_gt.map(x=>x.p)),class:'full'}));if(visible.gt&&matched.length)plot.append(E('path',{d:path(matched.map(x=>x.gt)),class:'gtPath'}));if(visible.pred)plot.append(E('path',{d:path(data.frames.map(x=>x.pred)),class:'predPath'}));
 const dl=extent*.07;data.frames.forEach(f=>{const p=project(f.pred),g=f.matched?project(f.gt):null;if(g&&visible.gt&&visible.pred)plot.append(E('line',{x1:g[0],y1:g[1],x2:p[0],y2:p[1],class:'connector'}));if(g&&visible.gt&&visible.dirs){const q=project(f.gt.map((v,k)=>v+f.gt_dir[k]*dl));plot.append(E('line',{x1:g[0],y1:g[1],x2:q[0],y2:q[1],class:'gtDir'}))}if(visible.pred&&visible.dirs){const q=project(f.pred.map((v,k)=>v+f.pred_dir[k]*dl));plot.append(E('line',{x1:p[0],y1:p[1],x2:q[0],y2:q[1],class:'predDir'}))}
  if(g&&visible.gt){const c=E('circle',{cx:g[0],cy:g[1],r:5,class:'gtPoint'+(selected===f.index?' selected':'')});c.onclick=()=>selectFrame(f);plot.append(c)}if(visible.pred){const c=E('polygon',{points:`${p[0]},${p[1]-6} ${p[0]-6},${p[1]+6} ${p[0]+6},${p[1]+6}`,class:'predPoint'+(selected===f.index?' selected':'')});c.onclick=()=>selectFrame(f);plot.append(c)}})}
document.querySelectorAll('[data-layer]').forEach(b=>b.onclick=()=>{let k=b.dataset.layer;visible[k]=!visible[k];b.setAttribute('aria-pressed',visible[k]);draw()});document.getElementById('reset').onclick=()=>{yaw=-.55;pitch=-.35;zoom=1;sceneYaw=-.55;scenePitch=-.3;sceneZoom=1;scenePanX=0;scenePanY=0;draw();renderScene()};plot.onpointerdown=e=>{drag=true;last=[e.clientX,e.clientY];plot.setPointerCapture(e.pointerId)};plot.onpointermove=e=>{if(!drag)return;yaw+=(e.clientX-last[0])*.009;pitch=Math.max(-1.45,Math.min(1.45,pitch+(e.clientY-last[1])*.009));last=[e.clientX,e.clientY];draw()};plot.onpointerup=()=>drag=false;plot.onwheel=e=>{e.preventDefault();zoom=Math.max(.45,Math.min(4,zoom*Math.exp(-e.deltaY*.001)));draw()};
document.getElementById('prevFrame').onclick=()=>stepFrame(-1);document.getElementById('nextFrame').onclick=()=>stepFrame(1);
document.getElementById('trajectoryMode').onclick=()=>setDisplayMode('trajectory');document.getElementById('sceneMode').onclick=()=>setDisplayMode('scene');document.getElementById('renderSelect').onchange=()=>{const f=data?.frames.find(x=>x.index===selected);if(f)selectFrame(f)};document.getElementById('cameraDisplay').onchange=renderScene;document.getElementById('pointBudget').onchange=()=>{sceneCloud=null;loadScene()};sceneCanvas.oncontextmenu=e=>e.preventDefault();sceneCanvas.onpointerdown=e=>{sceneDrag=true;scenePanDrag=e.shiftKey||e.button===1||e.button===2;sceneLast=[e.clientX,e.clientY];sceneCanvas.setPointerCapture(e.pointerId)};sceneCanvas.onpointermove=e=>{if(!sceneDrag)return;const dx=e.clientX-sceneLast[0],dy=e.clientY-sceneLast[1],dpr=Math.min(devicePixelRatio||1,1.5);if(scenePanDrag){scenePanX+=dx*dpr;scenePanY+=dy*dpr}else{sceneYaw+=dx*.009;scenePitch=Math.max(-1.5,Math.min(1.5,scenePitch+dy*.009))}sceneLast=[e.clientX,e.clientY];renderScene()};sceneCanvas.onpointerup=()=>sceneDrag=false;sceneCanvas.onwheel=e=>{e.preventDefault();const rect=sceneCanvas.getBoundingClientRect(),dpr=Math.min(devicePixelRatio||1,1.5),cx=(e.clientX-rect.left)*dpr-sceneCanvas.width/2,cy=(e.clientY-rect.top)*dpr-sceneCanvas.height/2,old=sceneZoom,next=Math.max(.1,Math.min(100,old*Math.exp(-e.deltaY*.0015))),ratio=next/old;scenePanX=cx-(cx-scenePanX)*ratio;scenePanY=cy-(cy-scenePanY)*ratio;sceneZoom=next;renderScene()};
function loadExperiment(id){fetch(`/api/data/${id}`).then(r=>{if(!r.ok)throw new Error('实验数据不存在');return r.json()}).then(d=>{data=d;selected=null;sceneCloud=null;document.getElementById('experiment').textContent=`${d.dataset} / ${d.scene} / ${d.views}视角 / ${d.method}`;let m=d.metrics,s=document.getElementById('stats');s.innerHTML=d.has_ground_truth?`<span>匹配 <b>${m.matched_frames}/${d.view_count}</b></span><span>ATE <b>${m.ate_rmse}</b></span><span>旋转 <b>${m.mean_rotation_error_deg}°</b></span>`:`<span class="warning">${d.reference_warning||'未找到 transforms.json，仅显示预测轨迹'}</span>`;document.getElementById('meanPos').textContent=m.mean_position_error!=null?m.mean_position_error.toFixed(4):'—';document.getElementById('ateValue').textContent=m.ate_rmse!=null?m.ate_rmse.toFixed(4):'—';document.getElementById('meanRot').textContent=m.mean_rotation_error_deg!=null?m.mean_rotation_error_deg.toFixed(2)+'°':'—';document.getElementById('gaussianFullCount').textContent=d.reconstruction.gaussian_count_full?.toLocaleString()||'—';document.getElementById('gaussianCount').textContent=d.reconstruction.gaussian_count?.toLocaleString()||'—';document.getElementById('pointCount').textContent=d.reconstruction.pointcloud_count?.toLocaleString()||'—';document.getElementById('displayedPointCount').textContent='—';populateFrameSelect();setRenderOptions();setVideo();if(displayMode==='scene')loadScene();else draw();if(d.frames.length)selectFrame(d.frames[0])}).catch(e=>{document.getElementById('errors').textContent=e})}
function options(id,values,label){const s=document.getElementById(id),old=s.value;s.replaceChildren();values.forEach(v=>{const o=document.createElement('option');o.value=v;o.textContent=label?label(v):v;s.append(o)});if(values.includes(old))s.value=old}
function refresh(level=0){const ds=document.getElementById('datasetSelect'),ss=document.getElementById('sceneSelect'),vs=document.getElementById('viewSelect'),ms=document.getElementById('methodSelect');if(level<=0)options('datasetSelect',[...new Set(viewerConfig.experiments.map(e=>e.dataset))],v=>viewerConfig.dataset_labels[v]||v);let list=viewerConfig.experiments.filter(e=>e.dataset===ds.value);if(level<=1)options('sceneSelect',[...new Set(list.map(e=>e.scene))],v=>v.split('_',1)[0]+' · '+v.slice(3,15));list=list.filter(e=>e.scene===ss.value);if(level<=2)options('viewSelect',[...new Set(list.map(e=>e.views))].sort((a,b)=>a-b),v=>v+' 视角');list=list.filter(e=>e.views===Number(vs.value));if(level<=3)options('methodSelect',[...new Set(list.map(e=>e.method))],v=>viewerConfig.method_labels[v]||v);const found=list.find(e=>e.method===ms.value);if(found)loadExperiment(found.id)}
let viewerConfig;fetch('/api/config').then(r=>r.json()).then(c=>{viewerConfig=c;document.getElementById('datasetSelect').onchange=()=>refresh(1);document.getElementById('sceneSelect').onchange=()=>refresh(2);document.getElementById('viewSelect').onchange=()=>refresh(3);document.getElementById('methodSelect').onchange=()=>refresh(4);refresh(0)});new ResizeObserver(()=>{draw();renderScene()}).observe(document.querySelector('.left'));
</script>
</body></html>"""


class ViewerHandler(BaseHTTPRequestHandler):
    payloads: dict[str, bytes]
    media: dict[str, dict[str, Path]]
    experiments: list[dict]

    def do_GET(self) -> None:  # noqa: N802
        path = unquote(urlparse(self.path).path)
        if path == "/":
            self._bytes(HTML.encode("utf-8"), "text/html; charset=utf-8")
            return
        if path == "/api/config":
            config = json.dumps(
                {"experiments": self.experiments,
                 "dataset_labels": {"random": "随机采样", "short60": "前60帧均匀采样"},
                 "method_labels": {"querysplat": "QuerySplat", "querysplat-tto20": "QuerySplat · TTO20", "zipsplat": "ZipSplat-QSPrior", "zipsplat-noprior": "ZipSplat-NoPrior · GT初始化位姿优化"}}
            ).encode("utf-8")
            self._bytes(config, "application/json; charset=utf-8")
            return
        match = re.fullmatch(r"/api/data/([^/]+)", path)
        if match:
            payload = self.payloads.get(match.group(1))
            if payload:
                self._bytes(payload, "application/json; charset=utf-8")
                return
        match = re.fullmatch(r"/media/([^/]+)/pointcloud", path)
        if match:
            media_path = self.media.get(match.group(1), {}).get("pointcloud")
            if media_path and media_path.is_file():
                self._bytes(media_path.read_bytes(), "application/octet-stream")
                return
        match = re.fullmatch(r"/media/([^/]+)/video/([^/]+)", path)
        if match:
            experiment_id, key = match.group(1), match.group(2)
            media_path = self.media.get(experiment_id, {}).get(f"video/{key}")
            if media_path and media_path.is_file():
                self._bytes(media_path.read_bytes(), "video/mp4")
                return
        match = re.fullmatch(r"/media/([^/]+)/(input|render|render_full|render_filtered|render_gt)/(\d+)", path)
        if match:
            experiment_id, kind, index = match.group(1), match.group(2), match.group(3)
            media_path = self.media.get(experiment_id, {}).get(f"{kind}/{index}")
            if media_path and media_path.is_file():
                self._bytes(media_path.read_bytes(), mimetypes.guess_type(media_path.name)[0] or "application/octet-stream")
                return
        self.send_error(404)

    def _bytes(self, content: bytes, content_type: str) -> None:
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(content)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(content)

    def log_message(self, format: str, *args: object) -> None:
        return


def main() -> None:
    args = parse_args()
    selected = args.experiment or choose_folder()
    selected = selected.expanduser().resolve()
    records: list[tuple[str, str, int, str, Path]] = []
    for dataset_dir in sorted(p for p in selected.iterdir() if p.is_dir()):
        for scene_dir in sorted(p for p in dataset_dir.iterdir() if p.is_dir()):
            for views_dir in sorted(p for p in scene_dir.iterdir() if p.is_dir()):
                match = re.fullmatch(r"(\d+)views", views_dir.name)
                if not match:
                    continue
                for method_dir in sorted(p for p in views_dir.iterdir() if p.is_dir()):
                    if (method_dir / "output/predicted_input_cameras.json").is_file():
                        records.append((dataset_dir.name, scene_dir.name, int(match.group(1)), method_dir.name, method_dir.resolve()))
    if not records:
        raise FileNotFoundError(f"No organized experiments found below {selected}")

    payloads: dict[str, bytes] = {}
    all_media: dict[str, dict[str, Path]] = {}
    experiment_config: list[dict] = []
    for index, (dataset, scene, views, method, experiment) in enumerate(records):
        experiment_id = f"e{index}"
        transforms = find_transforms(experiment, args.transforms)
        data, media = load_experiment(experiment, transforms)
        data.update({"selected_id": experiment_id, "dataset": dataset, "scene": scene, "views": views, "method": method})
        payloads[experiment_id] = json.dumps(data, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        all_media[experiment_id] = media
        experiment_config.append({"id": experiment_id, "dataset": dataset, "scene": scene, "views": views, "method": method})
        print(f"Indexed {dataset}/{scene}/{views}views/{method}")
    if args.inspect:
        return

    ViewerHandler.payloads = payloads
    ViewerHandler.media = all_media
    ViewerHandler.experiments = experiment_config
    server = ThreadingHTTPServer((args.host, args.port), ViewerHandler)
    url = f"http://{args.host}:{server.server_port}/"
    print(f"Viewer: {url}")
    print("Press Ctrl+C to stop.")
    if not args.no_browser:
        threading.Timer(0.4, webbrowser.open, args=(url,)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
