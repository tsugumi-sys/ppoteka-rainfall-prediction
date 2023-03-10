# Conda command
# Need to specify bash in order for conda activate to work.
SHELL=/bin/bash
# Note that the extra activate is needed to ensure that the activate floats env to the front of PATH
CONDA_ACTIVATE=source $$(conda info --base)/etc/profile.d/conda.sh ; conda activate ; conda activate

###
# Common parameters
###
CONDA_ENV_NAME = poteka-pipeline-pytorch
EXPERIMENT_NAME = Conv-vs-SA
MODEL_NAME = SAMSeq2Seq


###
# Scripts for executing pipeline.
###
.PHONY: train
train:
	$(CONDA_ACTIVATE) $(CONDA_ENV_NAME) && chmod +x scripts/train.sh && scripts/train.sh -e $(EXPERIMENT_NAME) -m $(MODEL_NAME)

TRAIN_RUN_ID = 8aba28989b544895b89938c5261c564e
.PHONY: evaluate
evaluate:
	$(CONDA_ACTIVATE) $(CONDA_ENV_NAME) &&  python run_only_evaluate.py evaluate.re_run.parent_run_id=$(TRAIN_RUN_ID)

.PHONY: train_and_evaluate
train_and_evaluate:
	$(CONDA_ACTIVATE) $(CONDA_ENV_NAME) && chmod +x scripts/train_and_evaluate.sh && scripts/train_and_evaluate.sh -e $(EXPERIMENT_NAME) -m $(MODEL_NAME)


.PHONY: train_all_models
train_all_models:
	$(CONDA_ACTIVATE) $(CONDA_ENV_NAME) && chmod +x scripts/train_all_models.sh && scripts/train_all_models.sh -e $(EXPERIMENT_NAME)

.PHONY: train_different_inputs
train_different_inputs:
	$(CONDA_ACTIVATE) $(CONDA_ENV_NAME) && chmod +x scripts/train_different_inputparams.sh \
		&& scripts/train_different_inputparams.sh -e $(EXPERIMENT_NAME) -m $(MODEL_NAME)

###
# scripts for test models.
# Run all processes with only 100 datasets.
###
.PHONY: test-run
test-train:
	$(CONDA_ACTIVATE) $(CONDA_ENV_NAME) && chmod +x scripts/test_run.sh  && scripts/test_run.sh -e test-run -m $(MODEL_NAME)	
	
.PHONY: test-all-models
test-all-models:
	$(CONDA_ACTIVATE) $(CONDA_ENV_NAME) && chmod +x scripts/test_all_models.sh && scripts/test_all_models.sh -e test-run

###
# Checking results via UI.
###
.PHONY: ui
ui:
	$(CONDA_ACTIVATE) $(CONDA_ENV_NAME) &&  mlflow ui -p 2345

###
# Unittests
###
.PHONY: test
test:
	$(CONDA_ACTIVATE) $(CONDA_ENV_NAME) &&  python -m unittest -v

###
# Build docs
###
PDOC_LOCATION=../docs/pdoc_contents
.PHONY: build_docs
build_docs:
	pdoc -o $(PDOC_LOCATION) -d google preprocess/ && \
	pdoc -o $(PDOC_LOCATION) -d google train/ && \
	pdoc -o $(PDOC_LOCATION) -d google evaluate/ && \
	pdoc -o $(PDOC_LOCATION) -d google common/


###
# Other commands
###
.PHONY: format
format:
	$(CONDA_ACTIVATE) $(CONDA_ENV_NAME) && black .

# DEV in poetry commands
.PHONY: poetry_train
poetry_train:
	poetry run mlflow run --experiment-name ${EXPERIMENT_NAME} --env-manager local \
		-P 'input_parameters=rain/temperature' -P use_dummy_data=true -P use_test_model=false .

.PHONY: poetry_ui
poetry_ui:
	poetry run mlflow ui

.PHONY: poetry_test
poetry_test:
	poetry run python -m unittest -v $(TARGET_MODULE)
