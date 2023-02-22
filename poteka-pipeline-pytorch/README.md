# POTEKA-PIPELINE-PYTORCH

This project is a ML pipeline for managing all steps (preprocessing P-POTEKA
data, training and evaluate pytorch models) of experiments.

See [API documents](https://tsugumi-sys.github.io/ppoteka-rainfall-prediction/)

## Overviews

### Flow of pipeline

1. **Preprocessing:** Split all P-POTEKA datasets into train, validation and
   evaluation datasets and store its metadata.
2. **Training:** Train the target model with training datasets, prepared in
   preprocess step and store the model's parameters and learning curves, e.t.c.
3. **Evaluation:** Evaluate the trained model using the evaluation dataset.

### Tools

- `mlflow`: Building pipeline and UI tool.
- `hydra`: Maning hyper parameters e.t.c.
- `torch`: Defining machine learning models.

### Models

All models are difined in `train/src/models`.

- `Seq2Seq`: The sequence to sequence model using ConvLSTM (Shi et al., 2015).
- `SASeq2Seq`: The sequence to sequence model using the base Self-Attention
  ConvLSTM (Lin et al., 2020).
- `SAMSeq2Seq`: The sequence to sequence model using Self-Attention ConvLSTM,
  which Self-Attention Memory module is applied (Lin et al., 2020).

### Other info

- Use `conda` environment for using `cartopy`, which is a library for
  visualizing geo data.
- `Adam` optimizer is used.
- `BCE loss` is used because the output is scaled to [0, 1].
- `RMSE` and `R2Score` is used for evaluation.
- `unittest` is used for unittests.

## Getting Started

### Setup

Modify `pipeline_root_dir_path` in `conf/config.yaml`

```yaml
// conf/config.yaml

...

###
# Path info
###
pipeline_root_dir_path: {Put your path. Note that this path should be pipeline (poteka-pipeline-pytorch) root, not project root.}

...

```

Create `secrets` directory and put `secret.yaml`.

```bash
mkdir conf/secrets && touch conf/secrets/secret.yaml
```
Put secret api token for notification services (If you don't use, put dummy string).
Notification services are used for notifying the end of pipeline.

```yaml
// conf/secrets/secret.yaml

notify_api_token: xxxx (<- put this line)

```

Place `train_dataset.csv` and `test_dataset.json` in `preprocess/src`. You can
create them automatically via the command in `../poteka-data/Makefile`.
See [README.md](https://github.com/tsugumi-sys/ppoteka-rainfall-prediction/tree/main/poteka-data#selecting-training-and-test-datasets).

```bash
cd ../poteka-data
make select_train_dataset && make select_test_dataset
```

Set `CONDA_ENV_NAME` and `EXPERIMENT_NAME` (mlflow experiment name) and 
`MODEL_NAME` (the target model) in `./Makefile`.

```Makefile
...
###
# Common parameters
###
CONDA_ENV_NAME = poteka-pipeline-pytorch
EXPERIMENT_NAME = Conv-vs-SA
MODEL_NAME = SAMSeq2Seq
...
```

Change parameters in `conf/` and run commands in `Makefile`.
