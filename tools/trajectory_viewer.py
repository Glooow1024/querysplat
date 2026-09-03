#!/usr/bin/env python3
"""Interactive QuerySplat trajectory and render viewer.

Run without arguments to choose an experiment folder with a native folder dialog:

    python tools/trajectory_viewer.py

Or pass a folder containing ``input/`` and ``output/`` directly:

    python tools/trajectory_viewer.py D:\results\01_scene\16views
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
    selected = filedialog.askdirectory(title="选择 QuerySplat 实验目录（例如 16views）")
    root.destroy()
    if not selected:
        raise SystemExit("No experiment folder selected.")
    return Path(selected)


def resolve_experiment(path: Path) -> Path:
    path = path.expanduser().resolve()
    candidates = [path]
    if path.name == "output":
        candidates.append(path.parent)
    candidates.extend(p for p in path.iterdir() if p.is_dir() and p.name.endswith("views"))
    valid = [p for p in candidates if (p / "output" / "predicted_input_cameras.json").is_file()]
    if not valid:
        raise FileNotFoundError(
            f"No predicted_input_cameras.json found below {path}. "
            "Choose a folder such as 16views containing input/ and output/."
        )
    if len(valid) > 1:
        names = ", ".join(p.name for p in valid)
        raise RuntimeError(f"Multiple view folders found ({names}); choose one view folder directly.")
    return valid[0]


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
    return next((p for p in candidates if p.is_file()), None)


def frame_name(value: str) -> str | None:
    match = FRAME_RE.search(Path(value).name)
    return match.group(0) if match else None


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


def load_experiment(experiment: Path, transforms_path: Path | None) -> tuple[dict, dict[str, Path]]:
    output = experiment / "output"
    prediction = json.loads((output / "predicted_input_cameras.json").read_text(encoding="utf-8"))
    predicted_frames = prediction.get("frames", [])
    if not predicted_frames:
        raise ValueError("predicted_input_cameras.json contains no frames.")

    inputs = indexed_files(output / "input_frames")
    renders = indexed_files(output / "rendered")
    media: dict[str, Path] = {}
    for index, path in inputs.items():
        media[f"input/{index}"] = path
    for index, path in renders.items():
        media[f"render/{index}"] = path

    result: dict = {
        "experiment": str(experiment),
        "view_count": len(predicted_frames),
        "has_ground_truth": False,
        "reference_path": str(transforms_path) if transforms_path else None,
        "full_gt": [],
        "frames": [],
        "metrics": {},
        "reference_warning": None,
    }
    predicted_by_name = {f["name"]: f for f in predicted_frames}

    def append_unaligned_predictions() -> None:
        for f in predicted_frames:
            result["frames"].append(
                {
                    "index": int(f["index"]),
                    "name": f["name"],
                    "pred": np.asarray(f["c2w"], dtype=float)[:3, 3].round(6).tolist(),
                    "pred_dir": np.asarray(f["c2w"], dtype=float)[:3, 2].round(6).tolist(),
                    "has_input": int(f["index"]) in inputs,
                    "has_render": int(f["index"]) in renders,
                }
            )

    if transforms_path is None:
        append_unaligned_predictions()
        return result, media

    reference = json.loads(transforms_path.read_text(encoding="utf-8"))
    reference_frames = reference.get("frames", [])
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
        for i, f in enumerate(reference_frames)
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
        item = {
            "index": index,
            "name": name,
            "has_input": index in inputs,
            "has_render": index in renders,
            "matched": name in aligned_by_name,
        }
        if name in aligned_by_name:
            aligned_position, aligned_direction, error, angle = aligned_by_name[name]
            gt_c2w = gt_by_name[name]
            item.update(
                {
                    "gt": gt_c2w[:3, 3].round(6).tolist(),
                    "pred": aligned_position.round(6).tolist(),
                    "gt_dir": (gt_c2w[:3, :3] @ GT_TO_OPENCV)[:, 2].round(6).tolist(),
                    "pred_dir": aligned_direction.round(6).tolist(),
                    "position_error": round(error, 6),
                    "rotation_error_deg": round(angle, 3),
                }
            )
        else:
            raw = np.asarray(f["c2w"], dtype=float)
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
        "mean_rotation_error_deg": round(float(np.mean(rotation_errors)), 3),
        "max_position_error": round(float(np.max(position_errors)), 6),
    }
    return result, media


HTML = r"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>QuerySplat 轨迹与渲染检查器</title>
<style>
:root{color-scheme:light dark;--bg:#f5f6f8;--panel:#fff;--fg:#172033;--muted:#687386;--line:#d9dee8;--gt:#2878d0;--pred:#e47728;--track:#8893a3;--bad:#c83e4d}
@media(prefers-color-scheme:dark){:root{--bg:#11151d;--panel:#1a202b;--fg:#edf2fa;--muted:#9ca8b9;--line:#364052;--gt:#62aaf3;--pred:#f5a052;--track:#778397;--bad:#ff6f7d}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font:14px/1.45 system-ui,-apple-system,"Segoe UI",sans-serif}
header{padding:12px 18px;border-bottom:1px solid var(--line);background:var(--panel);display:flex;align-items:center;gap:16px;flex-wrap:wrap}h1{font-size:17px;margin:0;font-weight:600}.meta{color:var(--muted);font-size:12px;overflow-wrap:anywhere}
.layout{display:grid;grid-template-columns:minmax(440px,1fr) minmax(330px,440px);gap:12px;padding:12px;height:calc(100vh - 65px);min-height:600px}.panel{background:var(--panel);border:1px solid var(--line);border-radius:10px;overflow:hidden}.left{display:flex;flex-direction:column}.toolbar{padding:9px 12px;border-bottom:1px solid var(--line);display:flex;gap:8px;align-items:center;flex-wrap:wrap}button{font:inherit;color:var(--fg);background:transparent;border:1px solid var(--line);border-radius:7px;padding:6px 9px;cursor:pointer}button[aria-pressed=true]{background:color-mix(in srgb,var(--gt) 13%,transparent);border-color:var(--gt)}button:hover{border-color:var(--muted)}.sw{display:inline-block;width:9px;height:9px;margin-right:5px}.sw.gt{background:var(--gt);border-radius:50%}.sw.pred{background:var(--pred);transform:rotate(45deg)}.sw.track{background:var(--track)}
#plot{width:100%;height:100%;min-height:400px;touch-action:none;cursor:grab}#plot:active{cursor:grabbing}.frame{fill:transparent;stroke:var(--line)}.grid{stroke:var(--line);opacity:.55}.axis{stroke:var(--muted)}.axisText{fill:var(--fg);font-size:12px}.full{fill:none;stroke:var(--track);stroke-width:1.4;opacity:.75}.gtPath{fill:none;stroke:var(--gt);stroke-width:2}.predPath{fill:none;stroke:var(--pred);stroke-width:2;stroke-dasharray:5 4}.connector{stroke:var(--line)}.gtPoint{fill:var(--gt);stroke:var(--panel);stroke-width:1.5;cursor:pointer}.predPoint{fill:var(--pred);stroke:var(--panel);stroke-width:1.5;cursor:pointer}.selected{stroke:var(--fg);stroke-width:3}.gtDir{stroke:var(--gt);stroke-width:1.5}.predDir{stroke:var(--pred);stroke-width:1.5;stroke-dasharray:3 2}
.right{display:flex;flex-direction:column;min-height:0}.details{padding:10px 12px;border-bottom:1px solid var(--line)}.frame-title{font-size:16px;font-weight:600}.errors{margin-top:4px;color:var(--muted)}.images{padding:10px;overflow:auto;display:grid;gap:12px}.image-block h2{font-size:13px;margin:0 0 5px;color:var(--muted);font-weight:500}.image-block img{display:block;width:100%;aspect-ratio:1;background:var(--bg);object-fit:contain;border:1px solid var(--line);border-radius:7px}.missing{width:100%;aspect-ratio:1;display:grid;place-items:center;border:1px dashed var(--line);border-radius:7px;color:var(--muted)}.stats{margin-left:auto;display:flex;gap:12px;color:var(--muted);font-size:12px}.stats b{color:var(--fg);font-weight:600}.warning{color:var(--bad)}
@media(max-width:850px){.layout{grid-template-columns:1fr;height:auto}.left{height:620px}.right{max-height:none}.stats{margin-left:0}.images{grid-template-columns:1fr 1fr}.layout{min-height:0}}
@media(max-width:560px){.layout{padding:6px}.left{height:520px}.images{grid-template-columns:1fr}}
</style>
</head>
<body>
<header><div><h1>QuerySplat 轨迹与渲染检查器</h1><div id="experiment" class="meta"></div></div><div id="stats" class="stats"></div></header>
<main class="layout">
 <section class="panel left">
  <div class="toolbar">
   <button data-layer="full" aria-pressed="true"><span class="sw track"></span>完整真值</button>
   <button data-layer="gt" aria-pressed="true"><span class="sw gt"></span>输入帧真值</button>
   <button data-layer="pred" aria-pressed="true"><span class="sw pred"></span>预测轨迹</button>
   <button data-layer="dirs" aria-pressed="true">朝向</button>
   <button id="reset">重置视角</button>
  </div>
  <svg id="plot" role="img" aria-label="可拖动的真值与预测相机轨迹"></svg>
 </section>
 <aside class="panel right">
  <div class="details"><div id="frameTitle" class="frame-title">选择一个相机节点</div><div id="errors" class="errors">点击蓝色圆点或橙色三角查看对应图像</div></div>
  <div class="images">
   <div class="image-block"><h2>真实输入（预处理后）</h2><div id="inputSlot"></div></div>
   <div class="image-block"><h2>QuerySplat 渲染结果</h2><div id="renderSlot"></div></div>
  </div>
 </aside>
</main>
<script>
const NS='http://www.w3.org/2000/svg',plot=document.getElementById('plot');
let data,yaw=-.55,pitch=-.35,zoom=1,drag=false,last,selected=null;
const visible={full:true,gt:true,pred:true,dirs:true};
const E=(tag,a={},text='')=>{const n=document.createElementNS(NS,tag);Object.entries(a).forEach(([k,v])=>n.setAttribute(k,v));if(text)n.textContent=text;return n};
const rotate=p=>{let[x,y,z]=p,cy=Math.cos(yaw),sy=Math.sin(yaw),cx=Math.cos(pitch),sx=Math.sin(pitch),xx=cy*x+sy*z,zz=-sy*x+cy*z;return[xx,cx*y-sx*zz,sx*y+cx*zz]};
function selectFrame(frame){selected=frame.index;document.getElementById('frameTitle').textContent=`${frame.name} · view ${frame.index}`;const e=document.getElementById('errors');e.textContent=frame.matched?`位置误差 ${frame.position_error.toFixed(3)} · 旋转误差 ${frame.rotation_error_deg.toFixed(2)}°`:'该帧没有对应的参考位姿';setImage('inputSlot',frame.has_input?`/media/input/${frame.index}`:null,'没有输入图片');setImage('renderSlot',frame.has_render?`/media/render/${frame.index}`:null,'没有渲染图片');draw()}
function setImage(id,url,message){const s=document.getElementById(id);s.replaceChildren();if(url){const im=document.createElement('img');im.src=url+'?v='+Date.now();im.alt=id==='inputSlot'?'真实输入图片':'QuerySplat 渲染结果';s.append(im)}else{const d=document.createElement('div');d.className='missing';d.textContent=message;s.append(d)}}
function draw(){if(!data)return;const w=Math.max(320,plot.clientWidth),h=Math.max(360,plot.clientHeight);plot.setAttribute('viewBox',`0 0 ${w} ${h}`);plot.replaceChildren();const pad=28,cx=w/2,cy=h/2;plot.append(E('rect',{x:pad,y:8,width:w-pad-8,height:h-20,class:'frame'}));for(let i=1;i<4;i++){let yy=8+i*(h-28)/4;plot.append(E('line',{x1:pad,y1:yy,x2:w-8,y2:yy,class:'grid'}))}
 const matched=data.frames.filter(f=>f.matched),all=[...data.full_gt.map(x=>x.p),...data.frames.map(x=>x.pred)];if(!all.length)return;const center=[0,1,2].map(k=>all.reduce((s,p)=>s+p[k],0)/all.length),rp=all.map(p=>rotate(p.map((v,k)=>v-center[k]))),ex=Math.max(...rp.map(p=>Math.abs(p[0])),1e-6),ey=Math.max(...rp.map(p=>Math.abs(p[1])),1e-6),scale=Math.min((w-pad-35)/(2*ex),(h-55)/(2*ey))*zoom,project=p=>{const q=rotate(p.map((v,k)=>v-center[k]));return[cx+q[0]*scale,cy-q[1]*scale,q[2]]},path=pts=>pts.map((p,i)=>{const q=project(p);return(i?'L':'M')+q[0].toFixed(1)+','+q[1].toFixed(1)}).join(' ');
 const o=project(center),extent=Math.max(...all.flatMap(p=>p.map((v,k)=>Math.abs(v-center[k]))));[['X',[extent*.15,0,0]],['Y',[0,extent*.15,0]],['Z',[0,0,extent*.15]]].forEach(([l,d])=>{const q=project(center.map((v,k)=>v+d[k]));plot.append(E('line',{x1:o[0],y1:o[1],x2:q[0],y2:q[1],class:'axis'}));plot.append(E('text',{x:q[0]+3,y:q[1]-3,class:'axisText'},l))});
 if(visible.full&&data.full_gt.length)plot.append(E('path',{d:path(data.full_gt.map(x=>x.p)),class:'full'}));if(visible.gt&&matched.length)plot.append(E('path',{d:path(matched.map(x=>x.gt)),class:'gtPath'}));if(visible.pred)plot.append(E('path',{d:path(data.frames.map(x=>x.pred)),class:'predPath'}));
 const dl=extent*.07;data.frames.forEach(f=>{const p=project(f.pred),g=f.matched?project(f.gt):null;if(g&&visible.gt&&visible.pred)plot.append(E('line',{x1:g[0],y1:g[1],x2:p[0],y2:p[1],class:'connector'}));if(g&&visible.gt&&visible.dirs){const q=project(f.gt.map((v,k)=>v+f.gt_dir[k]*dl));plot.append(E('line',{x1:g[0],y1:g[1],x2:q[0],y2:q[1],class:'gtDir'}))}if(visible.pred&&visible.dirs){const q=project(f.pred.map((v,k)=>v+f.pred_dir[k]*dl));plot.append(E('line',{x1:p[0],y1:p[1],x2:q[0],y2:q[1],class:'predDir'}))}
  if(g&&visible.gt){const c=E('circle',{cx:g[0],cy:g[1],r:5,class:'gtPoint'+(selected===f.index?' selected':'')});c.onclick=()=>selectFrame(f);plot.append(c)}if(visible.pred){const c=E('polygon',{points:`${p[0]},${p[1]-6} ${p[0]-6},${p[1]+6} ${p[0]+6},${p[1]+6}`,class:'predPoint'+(selected===f.index?' selected':'')});c.onclick=()=>selectFrame(f);plot.append(c)}})}
document.querySelectorAll('[data-layer]').forEach(b=>b.onclick=()=>{let k=b.dataset.layer;visible[k]=!visible[k];b.setAttribute('aria-pressed',visible[k]);draw()});document.getElementById('reset').onclick=()=>{yaw=-.55;pitch=-.35;zoom=1;draw()};plot.onpointerdown=e=>{drag=true;last=[e.clientX,e.clientY];plot.setPointerCapture(e.pointerId)};plot.onpointermove=e=>{if(!drag)return;yaw+=(e.clientX-last[0])*.009;pitch=Math.max(-1.45,Math.min(1.45,pitch+(e.clientY-last[1])*.009));last=[e.clientX,e.clientY];draw()};plot.onpointerup=()=>drag=false;plot.onwheel=e=>{e.preventDefault();zoom=Math.max(.45,Math.min(4,zoom*Math.exp(-e.deltaY*.001)));draw()};
fetch('/api/data').then(r=>r.json()).then(d=>{data=d;document.getElementById('experiment').textContent=d.experiment;let m=d.metrics,s=document.getElementById('stats');s.innerHTML=d.has_ground_truth?`<span>匹配 <b>${m.matched_frames}/${d.view_count}</b></span><span>ATE <b>${m.ate_rmse}</b></span><span>旋转 <b>${m.mean_rotation_error_deg}°</b></span>`:`<span class="warning">${d.reference_warning||'未找到 transforms.json，仅显示预测轨迹'}</span>`;draw();if(d.frames.length)selectFrame(d.frames[0])}).catch(e=>{document.getElementById('errors').textContent=e});new ResizeObserver(draw).observe(plot);
</script>
</body></html>"""


class ViewerHandler(BaseHTTPRequestHandler):
    payload: bytes
    media: dict[str, Path]

    def do_GET(self) -> None:  # noqa: N802
        path = unquote(urlparse(self.path).path)
        if path == "/":
            self._bytes(HTML.encode("utf-8"), "text/html; charset=utf-8")
            return
        if path == "/api/data":
            self._bytes(self.payload, "application/json; charset=utf-8")
            return
        if path.startswith("/media/"):
            key = path.removeprefix("/media/")
            media_path = self.media.get(key)
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
    experiment = resolve_experiment(selected)
    transforms = find_transforms(experiment, args.transforms)
    data, media = load_experiment(experiment, transforms)
    metrics = data["metrics"]
    print(f"Experiment: {experiment}")
    print(f"Views: {data['view_count']}; inputs: {sum(k.startswith('input/') for k in media)}; renders: {sum(k.startswith('render/') for k in media)}")
    if data["has_ground_truth"]:
        print(
            f"Reference: {transforms}; matched: {metrics['matched_frames']}; "
            f"ATE RMSE: {metrics['ate_rmse']}; rotation error: {metrics['mean_rotation_error_deg']} deg"
        )
    else:
        print(data["reference_warning"] or "Reference transforms.json not found; the viewer will show prediction only.")
    if args.inspect:
        return

    ViewerHandler.payload = json.dumps(data, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    ViewerHandler.media = media
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
