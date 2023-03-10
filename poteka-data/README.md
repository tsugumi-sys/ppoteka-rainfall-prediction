# PPOTEKA data cleaning and interpolation.

## Overview

Data cleaning process, interpolation process and selecting training & test
dataaset with commands in `Makefile`.

### Cleaning

#### Overview

cleaning has 3 processes.

1. **Raw data scaling and format datetime**: Read the data from the raw ppoteka
   data files and rescale each parameter values. And format datetime string.
   Finally, the cleaned data is saved into `data/cleaned-data/` directory.
2. **Imputation**: Data imputation is applied for missing values for the data in
   `$(DATA_DIR)/cleaned-data`. Finally inmputed data is saved into
   `data/imputed_data/` directory.
3. **Accumulation of rainfall data**: Hourly rainfall amount is calculated.
   Finally the data is saved into `data/accumulated-raf-data/` directory.
  
### Setups

#### 1. Change paraemters in `poteka-data/Makefile`

```bash
###
# Common parameters
###
CONDA_ENV_NAME=poteka <- Change your environment name.
N_CPUS=13             <- Change cpu counts for multiprocessing based on your PC specs.
TIME_STEP_MINUTES=10  <- Change time resolution of minute.
  
```

#### 2. Data Cleaning

Format timestamps and data imputation for data lacking, e.t.c. are executed.

Run the following command.

```bash
make data_cleaning
```

After cleaning step, the `data/` folder is as follows.

  ```
  .
├── data/
│   └── poteka-raw-data/
│   └── accumulated-raf-data/ (<- Created at cleaning step 3.)
│   └── cleaned-data/ (<- Created at cleaning step 1.)
│   └── imputed-data/ (<- Created at cleaning step 2.)
├── ...
  ```

#### 3. Interpolation

##### Overview

##### First step

Before interpolation, combining all observation points data of the same datetime
is created.

- **Creating `one-day-data`**: Reading from all the observation point data at the
  same datetime in `data/accumulated-raf-data`, then combine them into a single
  csv files.

##### Interpolation step.

Creating grid data from the observation points data. The interpolated grid data
and visualized map are saved in each weather parameter directories (e.g.
`data/rain_image/`, `data/temp_image/`).

#### 3-1. Create `one-day-data`.

Run the following commands as the first step (creating `one-data-data`).

```bash
make create_oneday_data
```

#### 3-2. Interpolation using `one-day-data`.

Then run the following commands for interpolation.

```bash
make interpolation
```

After interpolation step, the `data/` folder is as follows.

  ```
  .
├── data/
│   └── poteka-raw-data/
│   └── accumulated-raf-data/ 
│   └── cleaned-data/ 
│   └── imputed-data/
│   └── one-day-data/ (<- Greated at the first step of the interpolation process.)
│   └── rain_image/ (<- Created at the interpolation step.)
│   └── temp_image/ (<- Created at the interpolation step.)
│   └── ... (Other weather parameters' interpolated data directories)
├── ...
  ```

#### 4. Selecting training and test datasets.

##### Overview

1. Selecting test datasets. The test case meta file `test_dataset.json` are
   created and saved in `../poteka-pipeline-pytorch/preprocess/src/`. Modify
   `processing/select_test_dataset.py` for customing test cases.
2. Selecting training datasets. The training meta file `train_dataset.csv` are
   created and saved in `../poteka-pipeline-pytorch/preprocess/src/`. The data
   selected in test case is dropped so `test_dataset.json` is needed for this
   process.
  
#### 4-1. Prepare test-dataset.

```bash
make select_test_dataset
```

#### 4-2. Prepare train-dataset.

```bash
make select_train_dataset
```
   
After this process, new files are placed as follows.

  ```
  .
├── data/
├── poteka-pipeline-pytorch/
│   └── preprocess/
│       ├── src/
│       │   └── test_dataset.json (<- Placed here.)
│       │   └── train_dataset.csv (<- Placed here.)
├── ...
  ```

#### Other things

#### Directories

- `EDA/`: Exporatory data analysis for each weather parameters in jupyter notebooks.
