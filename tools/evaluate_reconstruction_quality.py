#!/usr/bin/env python3
"""Evaluate paired QuerySplat/ZipSplat input-view reconstruction quality."""

import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from skimage.metrics import structural_similarity
import lpips

OUT = Path("/root/multiview_compare/reports/baseline_all")
PAIRS = [
    ("random", Path("/root/multiview_compare/experiments/random")),
    ("short60", Path("/root/multiview_compare/experiments/short60")),
]


def ply_count(path):
    if not path.is_file(): return None
    with path.open("rb") as f: header = f.read(8192).decode("ascii", "ignore")
    import re
    m = re.search(r"element vertex (\d+)", header)
    return int(m.group(1)) if m else None


def reference_image(path):
    im = Image.open(path).convert("RGB")
    w, h = im.size; side = min(w, h)
    im = im.crop(((w-side)//2, (h-side)//2, (w+side)//2, (h+side)//2))
    return np.asarray(im.resize((256, 256), Image.Resampling.LANCZOS), np.float32) / 255


def rendered_image(path):
    return np.asarray(Image.open(path).convert("RGB").resize((256, 256), Image.Resampling.LANCZOS), np.float32) / 255


def write_csv(path, rows):
    if not rows: return
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def mean(values):
    vals = [float(x) for x in values if x is not None and math.isfinite(float(x))]
    return float(np.mean(vals)) if vals else None


def fmt(x, n=3): return "—" if x is None else f"{x:.{n}f}"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    metric = lpips.LPIPS(net="alex").to(device).eval()
    view_rows, experiment_rows = [], []
    for batch, dataset_root in PAIRS:
        for query_exp in sorted(dataset_root.glob("[0-9][0-9]_*/[0-9]*views/querysplat")):
            scene, views = query_exp.parents[1].name, int(query_exp.parent.name.removesuffix("views"))
            zip_exp = query_exp.parent / "zipsplat"
            if not (zip_exp / "output/gaussians.ply").is_file(): continue
            experiments = [("QuerySplat", query_exp), ("ZipSplat", zip_exp)]
            if batch == "short60":
                tto_exp = query_exp.parent / "querysplat-tto20"
                if (tto_exp / "output/gaussians_opacity0.ply").is_file():
                    experiments.append(("QuerySplat-TTO20", tto_exp))
            for model, exp in experiments:
                output = exp / "output"
                cameras = json.loads((output / "predicted_input_cameras.json").read_text())["frames"]
                refs, preds, metadata = [], [], []
                for frame in cameras:
                    idx, name = int(frame["index"]), frame["name"]
                    source = query_exp / "input" / f"{name}.png"
                    render = output / "rendered" / f"render_view{idx}.png"
                    if source.is_file() and render.is_file():
                        refs.append(reference_image(source)); preds.append(rendered_image(render)); metadata.append((idx, name))
                ref_t = torch.from_numpy(np.stack(refs)).permute(0,3,1,2).to(device) * 2 - 1
                pred_t = torch.from_numpy(np.stack(preds)).permute(0,3,1,2).to(device) * 2 - 1
                with torch.no_grad(): lp = metric(pred_t, ref_t).flatten().cpu().numpy()
                psnrs, ssims = [], []
                for i, ((idx, name), ref, pred) in enumerate(zip(metadata, refs, preds)):
                    mse = float(np.mean((pred-ref)**2)); psnr = -10*math.log10(max(mse, 1e-10))
                    ssim = float(structural_similarity(ref, pred, data_range=1, channel_axis=2))
                    psnrs.append(psnr); ssims.append(ssim)
                    view_rows.append({"batch":batch,"scene":scene,"views":views,"model":model,"view_index":idx,
                                      "frame":name,"psnr":psnr,"ssim":ssim,"lpips_alex":float(lp[i])})
                if model.startswith("QuerySplat"):
                    timing = json.loads((output / "inference_timing.json").read_text())
                    pose = timing.get("camera_and_depth_seconds"); recon = timing.get("image_to_3dgs_seconds"); render_t = timing.get("render_seconds")
                    full_count = ply_count(output / "gaussians_opacity0.ply")
                    retained_count = ply_count(output / "gaussians.ply")
                    if retained_count is None:
                        retained_count = ply_count(output / "gaussians_opacity0.05.ply")
                else:
                    timing = json.loads((output / "run_stats.json").read_text())
                    pose = 0.0; recon = timing.get("inference_seconds"); render_t = timing.get("render_seconds")
                    full_count = ply_count(output / "gaussians.ply"); retained_count = full_count
                experiment_rows.append({"batch":batch,"scene":scene,"views":views,"model":model,
                    "evaluated_images":len(refs),"mean_psnr":mean(psnrs),"mean_ssim":mean(ssims),"mean_lpips_alex":mean(lp),
                    "gaussians_full":full_count,"gaussians_saved_primary":retained_count,
                    "pose_depth_seconds":pose,"reconstruction_seconds":recon,"render_seconds":render_t,
                    "pipeline_seconds_excl_load":sum(x or 0 for x in (pose,recon,render_t)),
                    "gaussians_full_file_mib":(output/"gaussians_opacity0.ply").stat().st_size/2**20 if model.startswith("QuerySplat") else (output/"gaussians.ply").stat().st_size/2**20})
                print(batch, scene[:2], views, model, fmt(mean(psnrs)), fmt(mean(ssims)), fmt(mean(lp)), flush=True)

    aggregate_rows=[]
    groups=defaultdict(list)
    for row in experiment_rows: groups[(row["batch"],row["views"],row["model"])].append(row)
    for (batch,views,model), rows in sorted(groups.items()):
        aggregate_rows.append({"batch":batch,"views":views,"model":model,"experiments":len(rows),
            **{key:mean([r[key] for r in rows]) for key in ("mean_psnr","mean_ssim","mean_lpips_alex","gaussians_full","gaussians_saved_primary","pose_depth_seconds","reconstruction_seconds","render_seconds","pipeline_seconds_excl_load","gaussians_full_file_mib")}})
    write_csv(OUT/"per_view_metrics.csv",view_rows); write_csv(OUT/"per_experiment_metrics.csv",experiment_rows); write_csv(OUT/"aggregate_metrics.csv",aggregate_rows)
    (OUT/"results.json").write_text(json.dumps({"per_experiment":experiment_rows,"aggregate":aggregate_rows},indent=2)+"\n")

    lines=["# QuerySplat vs ZipSplat quantitative reconstruction report","",
      "## Protocol","", "- Input-view reconstruction: rendered image vs the corresponding source image.",
      "- Both reference and prediction are center-cropped/resized to 256x256.",
      "- LPIPS uses the standard AlexNet v0.1 model.",
      "- Times exclude one-off model loading. ZipSplat uses shared QuerySplat cameras, so its pose time is 0 here.",
      "- QuerySplat full Gaussian count comes from a new opacity-threshold-0 export; its primary PLY remains threshold 0.05.","",
      "## Aggregate results","",
      "| Batch | Views | Model | N | PSNR ↑ | SSIM ↑ | LPIPS ↓ | Full GS | Recon s ↓ | Render s ↓ | Pipeline s ↓ |","|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in aggregate_rows:
        lines.append(f"| {r['batch']} | {r['views']} | {r['model']} | {r['experiments']} | {fmt(r['mean_psnr'])} | {fmt(r['mean_ssim'],4)} | {fmt(r['mean_lpips_alex'],4)} | {r['gaussians_full']:.0f} | {fmt(r['reconstruction_seconds'])} | {fmt(r['render_seconds'])} | {fmt(r['pipeline_seconds_excl_load'])} |")
    lines += ["","## Interpretation cautions","", "- These are input-view reconstruction metrics, not held-out novel-view metrics.",
      "- ZipSplat is evaluated with QuerySplat-predicted shared cameras; the comparison isolates reconstruction/rendering more than camera estimation.",
      "- Runtime values come from each repository's own CUDA timing implementation and are useful operational measurements, not a kernel-level benchmark.",""]
    (OUT/"REPORT.md").write_text("\n".join(lines),encoding="utf-8")

if __name__ == "__main__": main()
