# POTEKA-PIPELINE-PYTORCH

This project is a ML pipeline for managing all steps (preprocessing P-POTEKA
data, training and evaluate pytorch models) of experiments.

See [API documents](https://tsugumi-sys.github.io/ppoteka-rainfall-prediction/)

## Flow of pipeline

1. **Preprocessing:** Split all P-POTEKA datasets into train, validation and
   evaluation datasets and store its metadata.
2. **Training:** Train the target model with training datasets, prepared in
   preprocess step and store the model's parameters and learning curves, e.t.c.
3. **Evaluation:** Evaluate the trained model using the evaluation dataset.

## Tools

- `mlflow`: Building pipeline and UI tool.
- `hydra`: Maning hyper parameters e.t.c.
- `torch`: Defining machine learning models.

## Models

All models are difined in `train/src/models`.

- `Seq2Seq`: The sequence to sequence model using ConvLSTM (Shi et al., 2015).
- `SASeq2Seq`: The sequence to sequence model using the base Self-Attention
  ConvLSTM (Lin et al., 2020).
- `SAMSeq2Seq`: The sequence to sequence model using Self-Attention ConvLSTM,
  which Self-Attention Memory module is applied (Lin et al., 2020).

## Other info

- Use `conda` environment for using `cartopy`, which is a library for
  visualizing geo data.
- `Adam` optimizer is used.
- `BCE loss` is used because the output is scaled to [0, 1].
- `RMSE` and `R2Score` is used for evaluation.
- `unittest` is used for unittests.

## Getting Started

### Setup

- Modify `project_root_dir_path` in `conf/config.yaml`
- Place `train_dataset.csv` and `test_dataset.json` in `preprocess/src`. You can
  create and save them via the command in `../poteka-data/Makefile`.
- Set `CONDA_ENV_NAME` and `EXPERIMENT_NAME` (mlflow experiment name) and
  `MODEL_NAME` (the target model).
- Modify parameters in `conf/`.
- Run commands in `Makefile`.
