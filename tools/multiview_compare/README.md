# Multi-model comparison toolkit

This is the version-controlled home of the QuerySplat/ZipSplat comparison
workflow. Experiment data, generated media, reports, and runtime logs remain
outside Git under `/root/multiview_compare` on the server.

## Source layout

```text
tools/multiview_compare/
├── evaluation/   # PSNR, SSIM, LPIPS, timing, and GS-count reports
├── pipelines/    # dataset preparation and QuerySplat/ZipSplat batch runs
├── rendering/    # predicted/GT trajectory and GS-variant rendering
├── viewer.py     # browser UI and read-only media server
├── run_viewer.sh
└── open_viewer_tunnel.bat
```

Scripts in `rendering/` share Gaussian loading and Sim(3) utilities from
`gaussian_io.py`. Server launchers call the checked-in scripts directly; they
do not depend on temporary copies in `/tmp`.

## Server layout

- Source: `/root/querysplat_ws/querysplat/tools/multiview_compare`
- Experiments: `/root/multiview_compare/experiments`
- Reports: `/root/multiview_compare/reports`
- Runtime log: `/root/multiview_compare/logs/viewer.log`

The experiment hierarchy and method naming rules are documented in
`RESULT_LAYOUT.md`.

## Start or restart the viewer

```bash
cd /root/querysplat_ws/querysplat
bash tools/multiview_compare/run_viewer.sh 18765
```

The launcher runs `viewer.py` directly from this Git checkout. There is no
second deployed copy of the application source under the results directory.

## Open from Windows

Run `open_viewer_tunnel.bat`, or create the tunnel manually:

```powershell
ssh -N -L 18765:127.0.0.1:18765 vllm1
```

Then visit <http://127.0.0.1:18765/>.

## Dependencies

The viewer requires Python 3 and NumPy. It reads experiment artifacts without
modifying them.
