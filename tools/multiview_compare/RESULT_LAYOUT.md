# Multi-model comparison result layout

All server-side experiment outputs use one canonical root:

```text
/root/multiview_compare/
├── experiments/<dataset>/<scene>/<N>views/<method>/
├── reports/<report-name>/
├── manifests/<dataset>/
├── inputs/<prepared-input-name>/
├── viewer/
└── logs/
```

Datasets currently are `random` and `short60`. Methods currently are
`querysplat`, `querysplat-tto20`, and `zipsplat`. A method directory owns its
`input/`, `output/`, and selection manifest. Method directories are real
directories rather than symbolic links.

Sampled `input/` images may remain symbolic links to the immutable DL3DV source
dataset to avoid duplicating source data. Generated results under `output/`
must be regular files/directories.

New scripts should accept or default to the canonical root and write directly
to the method directory. Do not recreate model-specific result roots below the
source repositories.
