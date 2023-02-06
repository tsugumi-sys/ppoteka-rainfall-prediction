# PPOTEKA Rainfall Prediction

This repository contains two parts for rainfall prediction using PPOTEKA data.

1. Data processing of the raw P-POTEKA data.
2. Training machine larning models for rainfall prediction.

## Getting started

1. Place the raw PPOTEKA dataset to `$(PROJECT_ROOT)/data/poteka-raw-data/`
2. Exacute data cleaning and interpolation in `poteka-data`.
3. Training and evaluation machine learning models in `poteka-pipeline-pytorch`.

#### NOTE

- Conda is used in this project just because the visualization library `cartopy`
  is only works in conda environment (pip doesn't works). If you want to use
  other virtual environment, you have to consider about alternatives of
  `cartopy`.
