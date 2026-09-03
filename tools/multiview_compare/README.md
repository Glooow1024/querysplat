# Multi-model comparison viewer

This directory is the version-controlled source for the QuerySplat/ZipSplat
comparison viewer. Experiment data and runtime logs remain outside Git under
`/root/multiview_compare` on the server.

## Server layout

- Source: `/root/querysplat_ws/querysplat/tools/multiview_compare`
- Experiments: `/root/multiview_compare/experiments`
- Reports: `/root/multiview_compare/reports`
- Runtime log: `/root/multiview_compare/logs/viewer.log`

The experiment hierarchy and method naming rules are documented in
`RESULT_LAYOUT.md`.

## Start or restart on the server

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
