#!/usr/bin/env python3
"""Evaluate the five short60 model/Gaussian-retention configurations."""

import csv, json, math
from collections import defaultdict
from pathlib import Path

import lpips, numpy as np, torch
from PIL import Image
from skimage.metrics import structural_similarity

ROOT = Path("/root/multiview_compare/experiments/short60")
OUT = Path("/root/multiview_compare/reports/short60_variants")


def ply_count(path):
    import re
    if not path.is_file(): return None
    header = path.open("rb").read(8192).decode("ascii", "ignore")
    match = re.search(r"element vertex (\d+)", header)
    return int(match.group(1)) if match else None


def image(path, reference=False):
    im = Image.open(path).convert("RGB")
    if reference:
        w, h = im.size; side = min(w, h)
        im = im.crop(((w-side)//2, (h-side)//2, (w+side)//2, (h+side)//2))
    return np.asarray(im.resize((256, 256), Image.Resampling.LANCZOS), np.float32) / 255


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)


def avg(rows, key):
    values = [float(row[key]) for row in rows if row.get(key) not in (None, "")]
    return float(np.mean(values)) if values else None


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    metric = lpips.LPIPS(net="alex").to(device).eval()
    per_view, per_experiment = [], []
    configs = [
        ("QuerySplat", "full", "querysplat", "rendered_full_gs", "gaussians_opacity0.ply"),
        ("QuerySplat", "filtered-0.05", "querysplat", "rendered_filtered_gs", "gaussians.ply"),
        ("QuerySplat-TTO20", "full", "querysplat-tto20", "rendered_full_gs", "gaussians_opacity0.ply"),
        ("QuerySplat-TTO20", "filtered-0.05", "querysplat-tto20", "rendered_filtered_gs", "gaussians_opacity0.05.ply"),
        ("ZipSplat", "native-full", "zipsplat", "rendered", "gaussians.ply"),
        ("ZipSplat-NoPrior", "native-full", "zipsplat-noprior", "rendered", "gaussians.ply"),
    ]
    for query_exp in sorted(ROOT.glob("[0-9][0-9]_*/[0-9]*views/querysplat")):
        scene, views = query_exp.parents[1].name, int(query_exp.parent.name.removesuffix("views"))
        for method, gs_variant, method_dir, render_dir, ply_name in configs:
            exp = query_exp.parent / method_dir; output = exp / "output"
            cameras = json.loads((output / "predicted_input_cameras.json").read_text())["frames"]
            refs, preds, meta = [], [], []
            for frame in cameras:
                idx, name = int(frame["index"]), frame["name"]
                source = query_exp / "input" / f"{name}.png"
                render = output / render_dir / f"render_view{idx}.png"
                if source.is_file() and render.is_file():
                    refs.append(image(source, True)); preds.append(image(render)); meta.append((idx, name))
            rt = torch.from_numpy(np.stack(refs)).permute(0,3,1,2).to(device)*2-1
            pt = torch.from_numpy(np.stack(preds)).permute(0,3,1,2).to(device)*2-1
            with torch.no_grad(): lp = metric(pt, rt).flatten().cpu().numpy()
            psnrs, ssims = [], []
            for i, ((idx, name), ref, pred) in enumerate(zip(meta, refs, preds)):
                mse=float(np.mean((pred-ref)**2)); psnr=-10*math.log10(max(mse,1e-10))
                ssim=float(structural_similarity(ref,pred,data_range=1,channel_axis=2))
                psnrs.append(psnr); ssims.append(ssim)
                per_view.append({"scene":scene,"views":views,"method":method,"gs_variant":gs_variant,"view_index":idx,"frame":name,"psnr":psnr,"ssim":ssim,"lpips_alex":float(lp[i])})
            if method.startswith("ZipSplat"):
                timing=json.loads((output/"run_stats.json").read_text()); pose=0.; recon=timing["inference_seconds"]; render_time=timing["render_seconds"]; registration=timing.get("posthoc_registration_seconds",0.) or 0.
            else:
                timing=json.loads((output/"inference_timing.json").read_text()); pose=timing["camera_and_depth_seconds"]; recon=timing["image_to_3dgs_seconds"]
                render_time=json.loads((output/render_dir/"render_manifest.json").read_text())["render_seconds"]; registration=0.
            per_experiment.append({"scene":scene,"views":views,"method":method,"gs_variant":gs_variant,"images":len(refs),"psnr":float(np.mean(psnrs)),"ssim":float(np.mean(ssims)),"lpips_alex":float(np.mean(lp)),"gaussians":ply_count(output/ply_name),"pose_depth_seconds":pose,"posthoc_registration_seconds":registration,"reconstruction_seconds":recon,"render_all_input_views_seconds":render_time})
            print(scene[:2],views,method,gs_variant,round(float(np.mean(psnrs)),3),flush=True)
    grouped=defaultdict(list)
    for row in per_experiment: grouped[(row["views"],row["method"],row["gs_variant"])].append(row)
    aggregate=[]
    for (views,method,variant),rows in sorted(grouped.items()):
        aggregate.append({"views":views,"method":method,"gs_variant":variant,"experiments":len(rows),**{k:avg(rows,k) for k in ("psnr","ssim","lpips_alex","gaussians","pose_depth_seconds","posthoc_registration_seconds","reconstruction_seconds","render_all_input_views_seconds")}})
    overall=[]
    for method,variant,_,_,_ in configs:
        rows=[r for r in per_view if r["method"]==method and r["gs_variant"]==variant]
        exps=[r for r in per_experiment if r["method"]==method and r["gs_variant"]==variant]
        overall.append({"method":method,"gs_variant":variant,"experiments":len(exps),"images":len(rows),"psnr":avg(rows,"psnr"),"ssim":avg(rows,"ssim"),"lpips_alex":avg(rows,"lpips_alex"),"gaussians":avg(exps,"gaussians"),"pose_depth_seconds":avg(exps,"pose_depth_seconds"),"posthoc_registration_seconds":avg(exps,"posthoc_registration_seconds"),"reconstruction_seconds":avg(exps,"reconstruction_seconds"),"render_all_input_views_seconds":avg(exps,"render_all_input_views_seconds")})
    write_csv(OUT/"overall.csv",overall); write_csv(OUT/"aggregate_by_views.csv",aggregate); write_csv(OUT/"per_experiment.csv",per_experiment); write_csv(OUT/"per_view.csv",per_view)
    (OUT/"results.json").write_text(json.dumps({"overall":overall,"by_views":aggregate,"per_experiment":per_experiment},indent=2)+"\n")
    lines=["# 前60帧重建配置对比报告","","## 评测矩阵","","共10个场景、4/8/12/16视角，每种配置40组、400张输入视角图像。QuerySplat两种方法分别比较全量GS与opacity≥0.05筛选GS；ZipSplat的原生PLY是完整模型输出。","","## 总体结果","","| 方法 | GS策略 | PSNR ↑ | SSIM ↑ | LPIPS ↓ | 平均GS数 | 重建s | 后置配准s | 全部输入视角渲染s |","|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for r in overall: lines.append(f"| {r['method']} | {r['gs_variant']} | {r['psnr']:.3f} | {r['ssim']:.4f} | {r['lpips_alex']:.4f} | {r['gaussians']:.0f} | {r['reconstruction_seconds']:.3f} | {r['posthoc_registration_seconds']:.3f} | {r['render_all_input_views_seconds']:.4f} |")
    lines += ["","## 分视角结果","","| 视角 | 方法 | GS策略 | PSNR ↑ | SSIM ↑ | LPIPS ↓ | GS数 | 重建s | 后置配准s | 渲染s |","|---:|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for r in aggregate: lines.append(f"| {r['views']} | {r['method']} | {r['gs_variant']} | {r['psnr']:.3f} | {r['ssim']:.4f} | {r['lpips_alex']:.4f} | {r['gaussians']:.0f} | {r['reconstruction_seconds']:.3f} | {r['posthoc_registration_seconds']:.3f} | {r['render_all_input_views_seconds']:.4f} |")
    lines += ["","## 口径说明","","- 图片指标是输入视角重建指标；真实图和预测图统一中心裁剪/缩放至256×256。","- QuerySplat全量与筛选结果均从对应PLY通过同一渲染器重新渲染，以避免把内存渲染和PLY重载混为一谈。","- QuerySplat同一方法的全量/筛选配置共享同一次重建，因此重建时间相同；差异在GS数、渲染耗时与图像结果。","- ZipSplat共享相机组在重建阶段使用QuerySplat预测相机先验。","- ZipSplat-NoPrior重建阶段只输入图像；完成重建后，仅用输入帧拟合一个共享Sim(3)，再转换同一套QuerySplat预测轨迹。它不使用DL3DV真实位姿，也不对每个评测视角单独优化。","- 后置配准时间独立于模型重建时间报告。","- LPIPS使用AlexNet v0.1；PSNR/SSIM越高越好，LPIPS越低越好。",""]
    (OUT/"REPORT_ZH.md").write_text("\n".join(lines),encoding="utf-8")


if __name__ == "__main__": main()
