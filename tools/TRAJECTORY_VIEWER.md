# QuerySplat trajectory viewer

This local viewer compares QuerySplat input-camera predictions with Nerfstudio/COLMAP reference poses and shows the corresponding input and rendered image when a camera node is clicked.

## Data layout

Select a view folder such as `16views`:

```text
01_scene_id/
├── transforms.json
└── 16views/
    ├── input/
    └── output/
        ├── predicted_input_cameras.json
        ├── input_frames/
        └── rendered/
```

`transforms.json` may also be supplied explicitly with `--transforms`.

## Windows

Install NumPy once if needed:

```powershell
py -m pip install numpy
```

Start with a native folder chooser:

```powershell
py tools\trajectory_viewer.py
```

Or pass the experiment directly:

```powershell
py tools\trajectory_viewer.py D:\querysplat-results\01_scene_id\16views
```

The viewer opens in the default browser. Drag to rotate, use the mouse wheel to zoom, and click a blue or orange camera node to inspect its input and render. Stop the local server with `Ctrl+C`.

## Validation only

```powershell
py tools\trajectory_viewer.py D:\querysplat-results\01_scene_id\16views --inspect
```

The reference and prediction are aligned with Sim(3), because monocular camera prediction has an arbitrary global coordinate frame and scale.
