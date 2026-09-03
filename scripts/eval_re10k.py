# Copyright (c) 2026 Inspatio.
# SPDX-License-Identifier: Apache-2.0
"""QuerySplat evaluation on RE10K, aligned with the C3G evaluation protocol.

Protocol (matches C3G eval_re10k.sh / src/evaluation):
  - Scenes + context/target splits from C3G's evaluation_index_re10k.json.
  - Context: index stores 2-view [left, right]; for num_context_views > 2 the extra
    views are evenly spaced between them (C3G add_addtional_context_index).
  - Target views are rendered and scored against GT (PSNR / SSIM / LPIPS).
  - Overlap tags: small (0.05-0.3), medium (0.3-0.55), large (0.55-0.8); rest ignored.
  - Images rescaled-and-center-cropped to the C3G eval resolution (224x224 for 2-view,
    256x256 for multi-view; C3G rescale_and_crop), same FOV for QuerySplat input and GT
    target, so metrics are comparable to C3G at the same resolution. QuerySplat renders
    at its native 512x512 then downsamples to the eval resolution for scoring.

Pose-free rendering (paper-faithful, no GT pose leakage into geometry):
  1. Context-only VGGT pass -> context cameras (frame F) + QuerySplat Gaussians (frame F).
  2. All-view VGGT pass (context+target) -> all cameras (frame F').
  3. Sim(3) align F' context centers -> F context centers -> (s, R, t).
  4. Apply (s, R, t) to F' target cameras -> target cameras in frame F.
  5. Render Gaussians at the aligned target cameras; compare to GT target images.

Only a configurable number of representative scenes save a side-by-side comparison
image (context | GT | pred | error map) to keep disk usage low.
"""

from __future__ import annotations

import argparse
import json
import os
from io import BytesIO
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from einops import rearrange
from PIL import Image
from safetensors.torch import load_file

from scripts.models import EncoderLatent, ModelInput, ModelInputDecoder, ModelInputEncoder, QuerySplat
from scripts.options import load_options_yaml
from scripts.utils.data import ImageTransform

RE10K_ORIG_SHAPE = (360, 640)  # H, W
# Eval resolution is chosen at runtime (224 for 2-view, 256 for multi-view), matching C3G.


# --------------------------- C3G-aligned helpers ---------------------------

def rescale_and_crop(images, intrinsics, shape):
    """C3G rescale_and_crop: scale so the smaller side matches, then center-crop.

    images: [*, 3, h_in, w_in] in [0,1]. intrinsics: [*, 3, 3] in PIXEL units of input.
    Returns cropped images and intrinsics (pixel units of the crop).
    """
    *_, h_in, w_in = images.shape
    h_out, w_out = shape
    scale_factor = max(h_out / h_in, w_out / w_in)
    h_scaled = round(h_in * scale_factor)
    w_scaled = round(w_in * scale_factor)
    *batch, c, h, w = images.shape
    images = images.reshape(-1, c, h, w)
    images = torch.stack([_rescale_one(img, (h_scaled, w_scaled)) for img in images])
    images = images.reshape(*batch, c, h_scaled, w_scaled)
    row = (h_scaled - h_out) // 2
    col = (w_scaled - w_out) // 2
    images = images[..., :, row:row + h_out, col:col + w_out]
    intrinsics = intrinsics.clone()
    intrinsics[..., 0, 0] *= w_in / w_out
    intrinsics[..., 1, 1] *= h_in / h_out
    intrinsics[..., 0, 2] -= col
    intrinsics[..., 1, 2] -= row
    return images, intrinsics


def _rescale_one(image, shape):
    h, w = shape
    img = (image * 255).clip(0, 255).to(torch.uint8)
    img = rearrange(img, "c h w -> h w c").detach().cpu().numpy()
    img = Image.fromarray(img).resize((w, h), Image.LANCZOS)
    img = np.array(img) / 255
    img = torch.tensor(img, dtype=image.dtype, device=image.device)
    return rearrange(img, "h w c -> c h w")


def get_overlap_tag(overlap):
    if 0.05 <= overlap <= 0.3:
        return "small"
    if overlap <= 0.55:
        return "medium"
    if overlap <= 0.8:
        return "large"
    return "ignore"


@torch.no_grad()
def compute_psnr(gt, pred):
    gt, pred = gt.clip(0, 1), pred.clip(0, 1)
    mse = ((gt - pred) ** 2).mean(dim=(1, 2, 3))
    return -10 * mse.log10()


@torch.no_grad()
def compute_ssim(gt, pred):
    from skimage.metrics import structural_similarity
    vals = [
        structural_similarity(
            g.detach().cpu().numpy(), p.detach().cpu().numpy(),
            win_size=11, gaussian_weights=True, channel_axis=0, data_range=1.0,
        )
        for g, p in zip(gt, pred)
    ]
    return torch.tensor(vals, dtype=pred.dtype, device=pred.device)


_LPIPS = None


@torch.no_grad()
def compute_lpips(gt, pred):
    global _LPIPS
    from lpips import LPIPS
    if _LPIPS is None:
        _LPIPS = LPIPS(net="vgg").to(pred.device)
    return _LPIPS.forward(gt, pred, normalize=True)[:, 0, 0, 0]


def sim3(source, target):
    """Return (scale, rotation, translation) mapping source points -> target points."""
    source_mean, target_mean = source.mean(0), target.mean(0)
    sc, tc = source - source_mean, target - target_mean
    cov = tc.T @ sc / len(source)
    u, singular, vt = np.linalg.svd(cov)
    sign = np.eye(3)
    sign[-1, -1] = np.sign(np.linalg.det(u @ vt))
    rotation = u @ sign @ vt
    variance = np.sum(sc * sc) / len(source)
    scale = float(np.trace(np.diag(singular) @ sign) / variance)
    translation = target_mean - scale * rotation @ source_mean
    return scale, rotation, translation


def apply_sim3_to_c2w(c2w, scale, rotation, translation):
    """Apply Sim(3) (s, R, t) to c2w: position <- s*R@p + t, rotation <- R@R."""
    R = torch.tensor(rotation, dtype=c2w.dtype, device=c2w.device)
    t = torch.tensor(translation, dtype=c2w.dtype, device=c2w.device)
    out = c2w.clone()
    out[..., :3, :3] = R @ out[..., :3, :3]
    out[..., :3, 3] = scale * (out[..., :3, 3] @ R.T) + t
    return out


def c2w_to_cam_view(c2w):
    """c2w [...,4,4] -> QuerySplat cam_view (transposed w2c) [...,4,4]."""
    w2c = torch.linalg.inv(c2w)
    return w2c.transpose(-2, -1)


def cam_view_to_c2w(cam_view):
    return torch.linalg.inv(cam_view.transpose(-2, -1))

# >>> PART2 <<<


# --------------------------- RE10K data ---------------------------

def load_index(path):
    with open(path) as f:
        return json.load(f)


def expand_ctx(pair, n):
    l, r = pair
    if n <= 2:
        return torch.tensor([l, r], dtype=torch.int64)
    return torch.linspace(l, r, n).long()


def convert_poses(poses):
    b, _ = poses.shape
    K = torch.eye(3, dtype=torch.float32).repeat(b, 1, 1)
    fx, fy, cx, cy = poses[:, :4].T
    K[:, 0, 0], K[:, 1, 1], K[:, 0, 2], K[:, 1, 2] = fx, fy, cx, cy
    w, h = RE10K[1], RE10K[0]
    K[:, 0, 0] *= w
    K[:, 1, 1] *= h
    K[:, 0, 2] *= w
    K[:, 1, 2] *= h
    w2c = torch.eye(4, dtype=torch.float32).repeat(b, 1, 1)
    w2c[:, :3] = rearrange(poses[:, 6:], "b (h w) -> b h w", h=3, w=4)
    return torch.linalg.inv(w2c), K


def decode_img(e):
    return Image.open(BytesIO(e.numpy().tobytes())).convert("RGB")


def iter_scenes(root, idx, n):
    td = Path(root) / "test"
    with (td / "index.json").open() as f:
        ci = json.load(f)
    items = list(idx.items())[:n] if n else list(idx.items())
    bc = {}
    for sc, e in items:
        if e is None:
            continue
        bc.setdefault(ci[sc], []).append((sc, e))
    for cn, es in bc.items():
        chunk = torch.load(td / cn, weights_only=False)
        ex = {x["key"]: x for x in chunk}
        for sc, e in es:
            if sc not in ex:
                print(f"Scene {sc} not in chunk {cn}; skipping.")
                continue
            yield sc, ex[sc], e


# --------------------------- QuerySplat inference ---------------------------

def load_model(cfg, ckpt, device):
    opt = load_options_yaml(cfg)
    m = QuerySplat(opt).to(device)
    st = load_file(ckpt, device="cpu")
    if len(st) != 521:
        raise ValueError(f"Expected 521 tensors, got {len(st)}")
    inc = m.load_state_dict(st, strict=False)
    if inc.unexpected_keys:
        raise ValueError(f"Unexpected keys: {inc.unexpected_keys}")
    m.eval()
    del st
    return m, opt


def prep_qs(img256):
    # img256: [V, 3, H, W] in [0,1]. ImageTransform.preprocess_images expects (N, C, H, W),
    # so pass img256 directly (V acts as batch). Official infer.py processes per-image (4D).
    t = ImageTransform(crop_size=(512, 512), sample_size=(512, 512), max_crop=True)
    out, _, _, _ = t.preprocess_images(img256)
    return out  # [V, 3, 512, 512]


def build_input(img512, device):
    n = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    enc = ModelInputEncoder(images_rgb=n(img512.clone()).unsqueeze(0),
                            images_rgb_unnormalized=img512.unsqueeze(0))
    dec = ModelInputDecoder(cam_view=torch.empty(0), intrinsics=torch.empty(0))
    return ModelInput(encoder=enc, decoder=dec)


def predict_cams(m, img512):
    with torch.no_grad():
        cv, K, _, _, _ = m.vggt_encoder.predict_cameras_and_depths(
            img512.unsqueeze(0), image_hw=m.img_size)
    return cv[0], K[0]


def forward_gaussians_with_cameras(model, encoder_input):
    """Like forward_gaussians, but also returns the context cameras/intrinsics.

    Uses a single VGGT pass so the cameras (frame F) are exactly the ones used to
    build the Plucker embeddings -> Gaussians (also frame F). Avoids a redundant
    second aggregator pass that predict_cameras_and_depths would otherwise need.
    """
    # NOTE: use no_grad (not inference_mode) so the returned cameras/gaussians are
    # normal tensors, not inference tensors. TTO needs to render through these
    # cameras with autograd enabled; inference tensors cannot be saved for backward.
    with torch.no_grad():
        vggt_output, cam_view, intrinsics, _ = model.vggt_encoder.forward_with_cameras(
            encoder_input.images_rgb_unnormalized, image_hw=model.img_size)
        keys, values = model.enc_dec_backbone._encode_to_kv(vggt_output.tokens, run_encoder=False)
        plucker = model._plucker_from_vggt_cameras(
            cam_view, intrinsics,
            dtype=encoder_input.images_rgb.dtype, device=encoder_input.images_rgb.device)
        rgb_tokens = model._original_encoder_tokens(encoder_input, plucker)
        post_rgb_keys, post_rgb_values = model._original_tokens_to_kv(rgb_tokens)
        latent = EncoderLatent(
            keys=keys, values=values, post_rgb_keys=post_rgb_keys,
            post_rgb_values=post_rgb_values, eye_token=vggt_output.eye_token,
            layer_weights=vggt_output.layer_weights)
        gaussians = model.forward_decoder(latent)
    return gaussians, cam_view[0], intrinsics[0]



# --------------------------- SE3 pose delta (mirrors C3G update_pose) ---------------------------

def _skew(x):
    s = torch.zeros(3, 3, device=x.device, dtype=x.dtype)
    s[0, 1] = -x[2]; s[0, 2] = x[1]; s[1, 0] = x[2]; s[1, 2] = -x[0]
    s[2, 0] = -x[1]; s[2, 1] = x[0]
    return s


def _so3_exp(theta):
    W = _skew(theta); W2 = W @ W
    angle = torch.norm(theta)
    I = torch.eye(3, device=theta.device, dtype=theta.dtype)
    if angle < 1e-5:
        return I + W + 0.5 * W2
    return I + (torch.sin(angle) / angle) * W + ((1 - torch.cos(angle)) / (angle ** 2)) * W2


def _se3_apply(tau, w2c):
    """Left-multiply SE3(tau) onto w2c, where tau = [trans, rot] (C3G convention)."""
    rho = tau[:3]; theta = tau[3:]
    R = _so3_exp(theta)
    W = _skew(theta); W2 = W @ W; angle = torch.norm(theta)
    I = torch.eye(3, device=tau.device, dtype=tau.dtype)
    if angle < 1e-5:
        V = I + 0.5 * W + (1.0 / 6.0) * W2
    else:
        V = I + W * ((1 - torch.cos(angle)) / (angle ** 2)) + W2 * ((angle - torch.sin(angle)) / (angle ** 3))
    T = torch.eye(4, device=tau.device, dtype=tau.dtype)
    T[:3, :3] = R; T[:3, 3] = V @ rho
    return T @ w2c


def align_pose(model, gaussians, c2w_init, K, gt_eval, device, n_steps=100,
               rot_lr=5e-3, trans_lr=5e-3):
    """C3G test_step_align: refine target poses with GT target images as supervision.

    Initial poses are the (Sim3-aligned) predicted target poses. Optimizes a
    left-multiplicative SE3 delta per view (mse + lpips loss), then renders.
    """
    n = c2w_init.shape[0]
    c2w = c2w_init.detach()
    rot_delta = torch.zeros(n, 3, device=device, requires_grad=True)
    trans_delta = torch.zeros(n, 3, device=device, requires_grad=True)
    opt = torch.optim.Adam([
        {"params": [rot_delta], "lr": rot_lr},
        {"params": [trans_delta], "lr": trans_lr},
    ])
    gt512 = F.interpolate(gt_eval, size=(512, 512), mode="bilinear", align_corners=False)
    for i in range(n_steps):
        opt.zero_grad()
        w2c = torch.linalg.inv(c2w)
        new_w2c = torch.stack([
            _se3_apply(torch.cat([trans_delta[j], rot_delta[j]], dim=-1), w2c[j]) for j in range(n)
        ])
        c2w_opt = torch.linalg.inv(new_w2c)
        dec = ModelInputDecoder(cam_view=c2w_to_cam_view(c2w_opt).unsqueeze(0), intrinsics=K.unsqueeze(0))
        rendered = model.render_gaussians(gaussians, dec)
        pred = rendered["images_pred"][0]
        loss = F.mse_loss(pred, gt512) + compute_lpips(gt512, pred).mean()
        loss.backward()
        opt.step()
    with torch.no_grad():
        w2c = torch.linalg.inv(c2w)
        new_w2c = torch.stack([
            _se3_apply(torch.cat([trans_delta[j], rot_delta[j]], dim=-1), w2c[j]) for j in range(n)
        ])
        c2w_opt = torch.linalg.inv(new_w2c)
        dec = ModelInputDecoder(cam_view=c2w_to_cam_view(c2w_opt).unsqueeze(0), intrinsics=K.unsqueeze(0))
        rendered = model.render_gaussians(gaussians, dec)
    return rendered, c2w_opt


class Acc:
    def __init__(self):
        self.a = None
        self.s = 0
        self.sub = {}
        self.ss = {}

    def update(self, mt, tag):
        if self.a is None:
            self.a = mt
            self.s = 1
        else:
            s = self.s
            self.a = {k: (s * v + mt[k]) / (s + 1) for k, v in self.a.items()}
            self.s += 1
        if tag and tag != "ignore":
            if tag not in self.sub:
                self.sub[tag] = mt
                self.ss[tag] = 1
            else:
                s = self.ss[tag]
                self.sub[tag] = {k: (s * v + mt[k]) / (s + 1) for k, v in self.sub[tag].items()}
                self.ss[tag] += 1

    def report(self, meth="ours"):
        from tabulate import tabulate
        ml = ["psnr", "lpips", "ssim"]
        rows = [(meth, *[f"{self.a[f'{m}_{meth}']:.3f}" for m in ml])]
        sec = ["All Pairs:", tabulate(rows, headers=["Method"] + ml)]
        for k, v in self.sub.items():
            rows = [(meth, *[f"{v[f'{m}_{meth}']:.3f}" for m in ml])]
            sec += [f"Overlap: {k}", tabulate(rows, headers=["Method"] + ml)]
        return "\n".join(sec)

    def json(self, meth="ours"):
        return {"num_scenes": self.s,
                "all_pairs": {k: float(v) for k, v in self.a.items()},
                **{f"overlap_{k}": {kk: float(vv) for kk, vv in v.items()} for k, v in self.sub.items()}}


def save_cmp(path, ctx, gt, pred):
    import torchvision.utils as vu
    err = (gt - pred.clamp(0, 1)).abs().mean(dim=1, keepdim=True).repeat(1, 3, 1, 1)
    vu.save_image(torch.cat([ctx, gt, pred, err], dim=0), str(path), nrow=gt.shape[0], padding=4)


def parse_args():
    p = argparse.ArgumentParser(description="QuerySplat RE10K evaluation (C3G-aligned)")
    p.add_argument("--config", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--re10k_root", required=True)
    p.add_argument("--index_path", required=True)
    p.add_argument("--num_context_views", type=int, default=12)
    p.add_argument("--eval_shape", type=int, default=0,
                   help="metric/render resolution (0=auto: 224 for 2-view, 256 for multi-view, matches C3G)")
    p.add_argument("--num_samples", type=int, default=1000)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--save_compare_n", type=int, default=10)
    p.add_argument("--use_tto", action="store_true")
    p.add_argument("--tto_n_steps", type=int, default=20)
    p.add_argument("--tto_lr", type=float, default=5e-3)
    p.add_argument("--tto_lpips_weight", type=float, default=0.05)
    p.add_argument("--align_pose", dest="align_pose", action="store_true", default=True,
                   help="C3G test_step_align: refine target poses with GT images (default on, matches C3G)")
    p.add_argument("--no_align_pose", dest="align_pose", action="store_false")
    p.add_argument("--pose_align_steps", type=int, default=100)
    p.add_argument("--rot_opt_lr", type=float, default=5e-3)
    p.add_argument("--trans_opt_lr", type=float, default=5e-3)
    return p.parse_args()


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    device = torch.device("cuda")
    os.makedirs(args.output_dir, exist_ok=True)
    idx = load_index(args.index_path)
    model, opt = load_model(args.config, args.checkpoint, device)
    opt = opt.evolve(num_input_views=args.num_context_views)
    # C3G uses 224x224 for 2-view, 256x256 for multi-view (re10k.yaml vs re10k_eval.yaml).
    if args.eval_shape == 0:
        eval_shape = (224, 224) if args.num_context_views <= 2 else (256, 256)
    else:
        eval_shape = (args.eval_shape, args.eval_shape)
    print(f"Eval resolution (metric + GT crop): {eval_shape[0]}x{eval_shape[1]}")
    acc = Acc()
    n = 0
    for scene, ex, entry in iter_scenes(args.re10k_root, idx, args.num_samples):
        raw_ov = entry.get("overlap")
        ov = raw_ov if isinstance(raw_ov, float) else (0.75 if raw_ov == "large" else 0.25)
        tag = get_overlap_tag(ov)
        if tag == "ignore":
            continue  # C3G skips ignore scenes (overlap > 0.8) entirely, incl. All Pairs. Note: overlap < 0.05 maps to "medium" in C3G's elif chain, not "ignore".
        ci = expand_ctx(entry["context"], args.num_context_views)
        ti = torch.tensor(entry["target"], dtype=torch.int64)
        # Pose-free: GT poses/intrinsics are unused (VGGT predicts cameras). Only images are needed.
        imgs = torch.stack([transforms.ToTensor()(decode_img(im)) for im in ex["images"]])
        ctx = imgs[ci]
        tgt = imgs[ti]
        ctx256, _ = rescale_and_crop(ctx, torch.eye(3).repeat(len(ci), 1, 1), eval_shape)
        tgt256, _ = rescale_and_crop(tgt, torch.eye(3).repeat(len(ti), 1, 1), eval_shape)
        ctx512 = prep_qs(ctx256.to(device))
        mi = build_input(ctx512, device)
        # 1. context-only pass -> Gaussians + context cameras in ONE consistent frame F
        gaussians, cv_ctx_F, K_ctx_F = forward_gaussians_with_cameras(model, mi.encoder)
        c2w_ctx_F = cam_view_to_c2w(cv_ctx_F)
        # 2. all-view pass -> all cameras (frame F')
        all256 = torch.cat([ctx256, tgt256], dim=0)
        all512 = prep_qs(all256.to(device))
        cv_all, K_all = predict_cams(model, all512)
        c2w_all_Fp = cam_view_to_c2w(cv_all)
        n_cv = len(ci)
        c2w_ctx_Fp = c2w_all_Fp[:n_cv]
        c2w_tgt_Fp = c2w_all_Fp[n_cv:]
        # 3. Sim(3) align F' context centers -> F context centers
        src = c2w_ctx_Fp[:, :3, 3].detach().cpu().numpy()
        dst = c2w_ctx_F[:, :3, 3].detach().cpu().numpy()
        s, R, t = sim3(src, dst)
        # 4. apply to F' target cameras -> frame F
        c2w_tgt_F = apply_sim3_to_c2w(c2w_tgt_Fp, s, R, t)
        K_tgt = K_all[n_cv:]
        # 5. optional TTO: reconstruct CONTEXT views (decoder=context cams, gt=context imgs)
        if args.use_tto:
            ctx_dec = ModelInputDecoder(cam_view=cv_ctx_F.unsqueeze(0),
                                        intrinsics=K_ctx_F.unsqueeze(0))
            mi_tto = ModelInput(encoder=mi.encoder, decoder=ctx_dec)
            gaussians = model.forward_tto_memory(mi_tto, n_steps=args.tto_n_steps,
                lr=args.tto_lr, lpips_weight=args.tto_lpips_weight, save_steps=[])
        # 6. optional align_pose (C3G test_step_align): refine target poses with GT images
        if args.align_pose:
            rendered, c2w_tgt_F = align_pose(model, gaussians, c2w_tgt_F, K_tgt,
                                              tgt256.to(device), device,
                                              n_steps=args.pose_align_steps,
                                              rot_lr=args.rot_opt_lr, trans_lr=args.trans_opt_lr)
        else:
            dec = ModelInputDecoder(cam_view=c2w_to_cam_view(c2w_tgt_F).unsqueeze(0),
                                    intrinsics=K_tgt.unsqueeze(0))
            with torch.inference_mode():
                rendered = model.render_gaussians(gaussians, dec)
        pred = rendered["images_pred"][0].clamp(0, 1)
        pred256 = torch.stack([_rescale_one(p, eval_shape) for p in pred])
        gt256 = tgt256.to(device)
        metrics = {
            "psnr_ours": compute_psnr(gt256, pred256).mean().item(),
            "ssim_ours": compute_ssim(gt256, pred256).mean().item(),
            "lpips_ours": compute_lpips(gt256, pred256).mean().item(),
        }
        acc.update(metrics, tag)
        n += 1
        if n % 50 == 0:
            print(f"[{n}] {acc.report()}")
        if args.save_compare_n and n <= args.save_compare_n:
            cmp_dir = Path(args.output_dir) / "comparisons"
            cmp_dir.mkdir(exist_ok=True)
            save_cmp(cmp_dir / f"{scene}.png", ctx256.cpu(), gt256.cpu(), pred256.cpu())
        del gaussians, rendered, mi
        torch.cuda.empty_cache()
    report = acc.report()
    (Path(args.output_dir) / "results.txt").write_text(report + "\n")
    (Path(args.output_dir) / "results.json").write_text(json.dumps(acc.json(), indent=2) + "\n")
    print("Final test results:\n" + report)
    print(f"Results saved to {args.output_dir}/results.txt and results.json")


if __name__ == "__main__":
    main()
