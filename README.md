# PPOTEKA Rainfall Prediction

This repository contains two parts for rainfall prediction using PPOTEKA data.

1. Data processing of the raw P-POTEKA data.
2. Training and evaluation of machine larning models for rainfall prediction.

## Getting started

1. Place the raw PPOTEKA dataset to `$(PROJECT_ROOT)/data/poteka-raw-data/`. The
   directory tree is as follows.

```bash
mkdir data && mkdir data/poteka-raw-data
```

```
.
├── data/ (<- You Need to Create This Folder.)
│   └── poteka-raw-data/ (<- You need to Create This Folder and place raw P-POTEKA data.)
│       ├── Anabu-1B_00181286/
│       │   └── ...
│       ├── ASTI_00173457/
│       │   └── ...
│       └── ...
├── docs/
├── poteka-data/
├── poteka-pipeline-pytorch/
└── README.md
```

2. Exacute data cleaning and interpolation in `poteka-data/`.

```bash
cd poteka-data && make data_cleaning && ... (See poteka-data/README.md)
```

3. Training and evaluation machine learning models in
   `poteka-pipeline-pytorch/`.

```bash
cd poteka-pipeline-pytorch && make ... (See poteka-pipeline-pytorch/README.md)
```

#### NOTE

- Conda is used in this project just because the visualization library `cartopy`
  is only works in conda environment (pip doesn't works). If you want to use
  other virtual environment, you have to consider about alternatives of
  `cartopy`.
