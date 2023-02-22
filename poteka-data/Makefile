# Need to specify bash in order for conda activate to work.
SHELL=/bin/bash
# Note that the extra activate is needed to ensure that the activate floats env to the front of PATH
CONDA_ACTIVATE=source $$(conda info --base)/etc/profile.d/conda.sh ; conda activate ; conda activate

###
# Common parameters
###
CONDA_ENV_NAME=p-poteka
DATA_DIR=../data
N_CPUS=13
TIME_STEP_MINUTES=10 

###
# Data cleaning
###
.PHONY: data_cleaning
data_cleaning:
	make extract_data && make impute_data && make calc_hourly_rain

extract_data:
	$(CONDA_ACTIVATE) $(CONDA_ENV_NAME) && \
	python processing/data-cleaning/extract-data.py \
		--data_dir_path=$(DATA_DIR) \
		--n_jobs=10 \
		--target_years=2019/2020

impute_data:
	$(CONDA_ACTIVATE) $(CONDA_ENV_NAME) && \
	python processing/data-cleaning/data-imputation.py \
		--data_dir_path=$(DATA_DIR) \
		--n_jobs=10
		
calc_hourly_rain:
	$(CONDA_ACTIVATE) $(CONDA_ENV_NAME) && \
	python processing/data-cleaning/accumulate-raf.py \
		--data_dir_path=$(DATA_DIR) \
		--n_jobs=10

###
# Interpolation
###
.PHONY: create_oneday_data
create_oneday_data:
	$(CONDA_ACTIVATE) $(CONDA_ENV_NAME) && python processing/data-making/make_oneday_data.py \
		--data_root_dir=$(DATA_DIR) \
		--n_jobs=$(N_CPUS) \
		--delta=$(TIME_STEP_MINUTES)

.PHONY: interpolation
interpolation:
	make preprocess_pressure_data && make preprocess_slp_data && \
	make preprocess_humidity_data && \
	make preprocess_temp_data && \
	make preprocess_wind_data && make preprocess_abs_wind_data && \
	make preprocess_rain_data

preprocess_pressure_data:
	$(CONDA_ACTIVATE) $(CONDA_ENV_NAME) && python processing/data-making/make_PRS_SLP_image.py \
		--data_root_path=$(DATA_DIR) \
		--target=prs \
		--n_jobs=$(N_CPUS) 

preprocess_slp_data: 
	$(CONDA_ACTIVATE) $(CONDA_ENV_NAME) && python processing/data-making/make_PRS_SLP_image.py \
		--data_root_path=$(DATA_DIR) \
		--target=slp \
		--n_jobs=$(N_CPUS) 

preprocess_humidity_data: 
	$(CONDA_ACTIVATE) $(CONDA_ENV_NAME) && python processing/data-making/make_humidity_image.py \
		--data_root_path=$(DATA_DIR) \
		--n_jobs=$(N_CPUS) 

preprocess_temp_data: 
	$(CONDA_ACTIVATE) $(CONDA_ENV_NAME) && python processing/data-making/make_temp_image.py \
		--data_root_path=$(DATA_DIR) \
		--n_jobs=$(N_CPUS) 

preprocess_wind_data: 
	$(CONDA_ACTIVATE) $(CONDA_ENV_NAME) && python processing/data-making/make_wind_image.py \
		--data_root_path=$(DATA_DIR) \
		--n_jobs=$(N_CPUS) \
		--target=uv

preprocess_abs_wind_data: 
	$(CONDA_ACTIVATE) $(CONDA_ENV_NAME) && python processing/data-making/make_wind_image.py \
		--data_root_path=$(DATA_DIR) \
		--n_jobs=$(N_CPUS) \
		--target=abs

preprocess_rain_data: 
	$(CONDA_ACTIVATE) $(CONDA_ENV_NAME) && python processing/data-making/make_rain_image.py \
		--data_root_path=$(DATA_DIR) \
		--time_step_minutes=$(TIME_STEP_MINUTES) \
		--n_jobs=$(N_CPUS)


###
# Select train dataset and test dataset,
###
select_train_dataset:
	python processing/data-making/select_train_dataset.py \
		--data_root_dir=$(DATA_DIR) \
		--max_rainfall_threshold=5

select_test_dataset:
	$(CONDA_ACTIVATE) $(CONDA_ENV_NAME) && python processing/data-making/select_test_dataset.py \
		--data_root_dir=$(DATA_DIR) \
		--timestep_delta=$(TIME_STEP_MINUTES)
