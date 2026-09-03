#!/usr/bin/env bash
set -euo pipefail

cd /root/zipsplat/ZipSplat
first="$(find /root/multiview_compare/experiments/random -mindepth 3 -maxdepth 3 -type d -name querysplat | sort | head -1)"
echo "TEST_DIR=$first"
find "$first/input" -maxdepth 1 -type l | sort
export D="$first/input"
export CUDA_VISIBLE_DEVICES=0
.venv/bin/python -c "import glob,time,torch,os; from zipsplat import ZipSplat,load_image; p=sorted(glob.glob(os.environ['D']+'/*.png')); print('images',len(p),p); t=time.time(); m=ZipSplat(weights='zipsplat').cuda().eval(); print('load_s',round(time.time()-t,3)); torch.cuda.reset_peak_memory_stats(); t=time.time(); g=m([load_image(x) for x in p],compression=1.0)[0]; torch.cuda.synchronize(); print('infer_s',round(time.time()-t,3),'shape',g.shape,'gaussians',g.num_gaussians,'peak_GiB',round(torch.cuda.max_memory_allocated()/2**30,3)); out='/root/zipsplat/ZipSplat/outputs/dl3dv_smoke_4views/scene.ply'; g.save_ply(out); print('saved',out,os.path.getsize(out))"
