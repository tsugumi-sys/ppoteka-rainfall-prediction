# PPOTEKA data cleaning and interpolation.

## Overview

Data cleaning process, interpolation process and selecting training & test
dataaset with commands in `Makefile`.

### Cleaning

cleaning has 3 processes.

1. **Raw data scaling and format datetime**: Read the data from the raw ppoteka
   data files and rescale each parameter values. And format datetime string.
   Finally, the cleaned data is saved into `data/cleaned-data/` directory.
2. **Imputation**: Data imputation is applied for missing values for the data in
   `$(DATA_DIR)/cleaned-data`. Finally inmputed data is saved into
   `data/imputed_data/` directory.
3. **Accumulation of rainfall data**: Hourly rainfall amount is calculated.
   Finally the data is saved into `data/accumulated-raf-data/` directory.

### Interpolation

#### First step

Before interpolation, combining all observation points data of the same datetime
is created.

- **Creaing `one-day-data`**: Reading from all the observation point data at the
  same datetime in `data/accumulated-raf-data`, then combine them into a single
  csv files.

#### Interpolation step.

Creating grid data from the observation points data. The interpolated grid data
and visualized map are saved in each weather parameter directories (e.g.
data/rain_image/, data/temp_image/).

### Selecting training and test datasets.

1. Selecting test datasets. The test case meta file `test_dataset.json` are
   created and saved in `../poteka-pipeline-pytorch/preprocess/src/`. Modify
   `processing/select_test_dataset.py` for customing test cases.
2. Selecting training datasets. The training meta file `train_dataset.csv` are
   created and saved in `../poteka-pipeline-pytorch/preprocess/src/`. The data
   selected in test case is dropped so `test_dataset.json` is needed for this
   process.
