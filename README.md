# PPOTEKA Rainfall Prediction

This repository contains two parts for rainfall prediction using PPOTEKA data.

1. Data processing of the raw P-POTEKA data.
2. Training and evaluation of machine larning models for rainfall prediction.

## Getting started

0. Create virtual environment of pyhton.

#### NOTE

- `Conda` is used in this project just because the visualization library `cartopy`
  is only works in conda environment (pip doesn't works). If you want to use
  other virtual environment, you have to consider about alternatives of
  `cartopy`.

[See installation guide.](https://docs.conda.io/projects/conda/en/latest/user-guide/install/linux.html)

Then, create virtualenv with conda like following command (See [official docs](https://docs.conda.io/projects/conda/en/latest/user-guide/getting-started.html) for more info).

```bash
conda create --name poteka python=3.9
```
(`poteka` is a example of the virtual environment's name. Change to your favorite name.)

After that, activate virtual env with following command:

```bash
conda activate poteka
  
```

Finally, install packages in `poteka-data/pyproject.toml` & `poteka-pipeline-pytorch/pyproject.toml` with conda.

Basically, you can install the libraries with `conda-forge` channel like the following command:

```bash
conda install -c conda-forge pandas
  
```

However, don't forget to see installation commands for each libralies' in official documentation.
 
1. Place the raw PPOTEKA dataset to `$(PROJECT_ROOT)/data/poteka-raw-data/`.
  The directory tree is as follows.
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
