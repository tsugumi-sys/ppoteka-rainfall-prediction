import json
import logging
import os
import sys
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from omegaconf import DictConfig
from sklearn.metrics import mean_squared_error, r2_score
from torch import nn

sys.path.append("..")
from common.config import (  # noqa: E402
    DEVICE,  # noqa: E402
    GridSize,
    MinMaxScalingValue,
    PPOTEKACols,
    ScalingMethod,  # noqa: E402
)
from common.custom_logger import CustomLogger  # noqa: E402
from common.utils import get_ob_point_values_from_tensor, rescale_tensor, timestep_csv_names  # noqa: E402
from evaluate.src.create_image import all_cases_scatter_plot, casetype_scatter_plot, date_scatter_plot  # noqa: E402
from evaluate.src.geoimg_generator.geoimg_generator_interactor import GeoimgGenratorInteractor  # noqa: E402
from evaluate.src.interpolator.interpolator_interactor import InterpolatorInteractor  # noqa: E402
from evaluate.src.utils import normalize_tensor, save_parquet  # noqa: E402

logger = CustomLogger(__name__)
logging.basicConfig(level=logging.INFO)


class BaseEvaluator:
    """Common methods for evaluation step."""

    def __init__(
        self,
        model: nn.Module,
        model_name: str,
        test_dataset: Dict,
        input_parameter_names: List[str],
        output_parameter_names: List[str],
        downstream_directory: str,
        observation_point_file_path: str,
        hydra_cfg: DictConfig,
    ) -> None:
        """

        Args:
            model (nn.Module): target model.
            model_name (str): target model name.
            test_dataset (Dict): Test dataset information like the following ...
                test_dataset: Dict
                { sample1: {
                    date: str,
                    start: str,
                    input: Tensor shape is (batch_size, num_channels, seq_len, height, width),
                    label: Tensor shape is (batch_size, num_channels, seq_len, height, width),
                    label_df: Dict[int, pd.DataFrame]
                    standarize_info: {"rain": {"mean": 1.0, "std": 0.3}, ...}
                 },
                 sample2: {...}
                }
            input_parameter_names (List[str]): Input parameter names.
            output_parameter_names (List[str]): Output parameter names.
            downstream_directory (str): Downstream directory path.
            observation_point_file_path (str): A observation point file path (observation_point.json)
            hydra_overrides (List[str]): Override information of hydra configurations.
        """
        self.model = model
        self.model_name = model_name
        self.test_dataset = test_dataset
        self.input_parameter_names = input_parameter_names
        self.output_parameter_names = output_parameter_names
        self.downstream_direcotry = downstream_directory
        self.observation_point_file_path = observation_point_file_path
        self.results_df = pd.DataFrame()
        self.metrics_df = pd.DataFrame()
        self.hydra_cfg = hydra_cfg

    def clean_dfs(self):
        self.results_df = pd.DataFrame()
        self.metrics_df = pd.DataFrame()

    def load_test_case_dataset(self, test_case_name: str) -> Tuple[torch.Tensor, torch.Tensor]:
        X_test = self.test_dataset[test_case_name]["input"].to(DEVICE)
        y_test = self.test_dataset[test_case_name]["label"].to(DEVICE)
        return X_test, y_test

    def rescale_pred_tensor(self, target_tensor: torch.Tensor, target_param: str) -> torch.Tensor:
        """This function rescale target_tensor to target parameter's original scale.
        e.g. rain tensor [0, 1] -> [0, 100]
        """
        if target_tensor.max().item() > 1 or target_tensor.min().item() < 0:
            raise ValueError(
                "Invalid scale of target tensor "
                f"(max: {target_tensor.max().item()}, min: {target_tensor.min().item()}). Should be scaled to [0, 1]"
            )

        target_tensor = normalize_tensor(target_tensor, DEVICE)
        min_val, max_val = MinMaxScalingValue.get_minmax_values_by_weather_param(target_param)
        rescaled_tensor = rescale_tensor(min_value=min_val, max_value=max_val, tensor=target_tensor.cpu().detach())
        return rescaled_tensor

    def add_result_df_from_pred_tensor(
        self,
        test_case_name: str,
        time_step: int,
        pred_tensor: torch.Tensor,
        label_df: pd.DataFrame,
        target_param: str,
    ) -> None:
        """This function is a interface for add result_df to self.result_df.

        Args:
            test_case_name (str):
            time_step (int):
            pred_tensor (torch.Tensor): A prediction tensor of a certain test case.
                This tensor should be scaled to its original scale.
            label_df (pd.DataFrame): A pandas dataframe of observation data.
            target_param (str):
        """
        pred_df = self.get_pred_df_from_tensor(
            pred_tensor
        )  # This dataframe has columns ["Pred_Value"] and index is observation names.
        result_df = label_df.merge(pred_df, right_index=True, left_index=True)
        result_df["test_case_name"] = test_case_name
        result_df["date"] = self.test_dataset[test_case_name]["date"]
        result_df["predict_utc_time"] = self.get_prediction_utc_time(test_case_name, time_step)
        result_df["target_parameter"] = target_param
        result_df["time_step"] = time_step
        result_df["case_type"] = "tc" if test_case_name.startswith("TC") else "not_tc"
        self.results_df = pd.concat([self.results_df, result_df], axis=0)

    def add_metrics_df_from_pred_tensor(
        self,
        test_case_name: str,
        time_step: int,
        pred_tensor: torch.Tensor,
        label_df: pd.DataFrame,
        target_param: str,
    ):
        """This function is a interface to add metrics_df from pred_tensor and label_df

        Args:
            test_case_name (str):
            time_step (int):
            pred_tensor (torch.Tensor):
            label_df (pd.DataFrame):
            target_param (str):
        """
        rmse_score = self.rmse_from_pred_tensor(pred_tensor, label_df, target_param)
        r2_score = self.r2_score_from_pred_tensor(pred_tensor, label_df, target_param)
        pred_utc_time = self.get_prediction_utc_time(test_case_name, time_step)

        metrics_df = pd.DataFrame(
            {
                "test_case_name": [test_case_name],
                "predict_utc_time": [pred_utc_time],
                "target_parameter": [target_param],
                "r2": [r2_score],
                "rmse": [rmse_score],
            }
        )
        self.metrics_df = pd.concat([self.metrics_df, metrics_df])

    def get_prediction_utc_time(self, test_case_name: str, time_step: int) -> str:
        """This function get UTC time of a prediction time of given test case and time step.

        Args:
            test_case_name (str): A test case name.
            time_step (int): A prediction time step from start time of a given test case.
        """
        _time_step_csvnames = timestep_csv_names(time_step_minutes=self.hydra_cfg.preprocess.time_step_minutes)
        predict_start = self.test_dataset[test_case_name]["start"]  # "hh-mm.csv"
        predict_start_idx = _time_step_csvnames.index(predict_start)

        utc_time_idx = predict_start_idx + time_step

        if utc_time_idx > len(_time_step_csvnames) - 1:
            utc_time_idx -= len(_time_step_csvnames)

        return _time_step_csvnames[utc_time_idx].replace(".csv", "")

    def calc_rmse(self, actual_ndarray: np.ndarray, target_ndarray: np.ndarray) -> float:
        return mean_squared_error(actual_ndarray, target_ndarray, squared=False)

    def calc_r2_score(self, actual_ndarray: np.ndarray, target_ndarray: np.ndarray) -> float:
        return r2_score(actual_ndarray, target_ndarray)

    def rmse_from_pred_tensor(self, pred_tensor: torch.Tensor, label_df: pd.DataFrame, target_param: str) -> float:
        """This function return rmse value between prediction and observation values.

        Args:
            pred_tensor (torch.Tensor): A prediction tensor of a certain test case.
                This tensor should be scaled to its original scale.
            label_df (pd.DataFrame): A pandas dataframe of observation data.
            target_param (str): A target weather parameter name.
        """
        # Pred_Value contains prediction values of each observation points.
        pred_df = self.get_pred_df_from_tensor(pred_tensor)
        check_df = pred_df.merge(label_df, right_index=True, left_index=True)
        rmse = self.calc_rmse(
            np.ravel(check_df[PPOTEKACols.get_col_from_weather_param(target_param)].astype(float).to_numpy()),
            np.ravel(check_df["Pred_Value"].astype(float).to_numpy()),
        )
        return rmse

    def rmse_from_results_df(
        self,
        output_param_name: str,
        target_date: Optional[str] = None,
        is_tc_case: Optional[bool] = None,
        target_time_steps: Optional[list[int]] = None,
    ) -> float:
        """This function calculate r2 score from results_df. Querying results_df with date and case_type:

        Args:
            output_param_name (str): A output weather parameter.
            target_date (Optional[str]): A target date for querying results_df.
            is_tc_case (Optional[bool]): A tc or not tc case type for querying results_df.

        """
        target_poteka_col = PPOTEKACols.get_col_from_weather_param(output_param_name)

        df = self.query_result_df(target_date=target_date, is_tc_case=is_tc_case, target_time_steps=target_time_steps)
        rmse = self.calc_rmse(
            np.ravel(df[target_poteka_col].astype(float).to_numpy()),
            np.ravel(df["Pred_Value"].astype(float).to_numpy()),
        )

        return rmse

    def r2_score_from_pred_tensor(self, pred_tensor: torch.Tensor, label_df: pd.DataFrame, target_param: str) -> float:
        """This funtion return r2 score between prediction and observation values.

        Args:
            pred_tensor (torch.Tensor): A prediction tensor of a certain test case.
                This tensor should be scaled to its original scale.
            label_df (pd.DataFrame): A pandas dataframe of observation data.
            target_param (str): A target weather parameter name.
        """
        pred_df = self.get_pred_df_from_tensor(pred_tensor)
        check_df = pred_df.merge(label_df, right_index=True, left_index=True)
        r2_score_val = self.calc_r2_score(
            np.ravel(check_df[PPOTEKACols.get_col_from_weather_param(target_param)].astype(float).to_numpy()),
            np.ravel(check_df["Pred_Value"].astype(float).to_numpy()),
        )
        return r2_score_val

    def r2_score_from_results_df(
        self,
        output_param_name: str,
        target_date: Optional[str] = None,
        is_tc_case: Optional[bool] = None,
        target_time_steps: Optional[list[int]] = None,
    ) -> float:
        """This function calculate r2 score from results_df. Querying results_df with date and case_type:

        Args:
            output_param_name (str): A output weather parameter.
            target_date (Optional[str]): A target date for querying results_df.
            is_tc_case (Optional[bool]): A tc or not tc case type for querying results_df.

        """
        target_poteka_col = PPOTEKACols.get_col_from_weather_param(output_param_name)

        df = self.query_result_df(target_date=target_date, is_tc_case=is_tc_case, target_time_steps=target_time_steps)

        r2_score_value = self.calc_r2_score(
            np.ravel(df[target_poteka_col].astype(float).to_numpy()),
            np.ravel(df["Pred_Value"].astype(float).to_numpy()),
        )
        return r2_score_value

    def query_result_df(
        self,
        target_date: Optional[str] = None,
        is_tc_case: Optional[bool] = None,
        target_time_steps: Optional[list[int]] = None,
    ):
        "This function get results_df queried with target date and is_tc_case flag."
        df = self.results_df.copy()
        if target_date is not None:
            query = [target_date in d for d in self.results_df["date"]]
            df = df.loc[query]

        if is_tc_case is not None:
            if is_tc_case is True:
                query = df["case_type"] == "tc"
            else:
                query = df["case_type"] == "not_tc"
            df = df.loc[query]

        if target_time_steps is not None:
            query = df["time_step"].isin(target_time_steps)
            df = df.loc[query]

        return df

    def get_pred_df_from_tensor(self, pred_tensor: torch.Tensor) -> pd.DataFrame:
        """This function generates prediction dataframe from prediction tensor.

        Return (pd.DataFrame): A prediction dataframe (columns: [`Pred_Value`], index: obsevation points name).
        """
        if pred_tensor.ndim > 2:
            raise ValueError(
                f"Invalid tensor dimentions for pred_tensor. The shape shold be less than 2, but {pred_tensor.shape}."
            )

        if pred_tensor.shape == torch.Size([GridSize.WIDTH, GridSize.HEIGHT]):
            pred_ob_point_tensor = get_ob_point_values_from_tensor(pred_tensor, self.observation_point_file_path)
        elif pred_tensor.shape == torch.Size([35]):
            pred_ob_point_tensor = pred_tensor
        else:
            raise ValueError(f"Invalid predict tensor shape.: {pred_tensor.shape}")

        with open(self.observation_point_file_path, "r") as f:
            ob_point_data = json.load(f)

        pred_df = pd.DataFrame(index=list(ob_point_data.keys()))
        pred_df["Pred_Value"] = (
            pred_ob_point_tensor.clone().detach().numpy().copy()
        )  # torch.float convert to numpy.float32
        return pred_df

    def save_results_df_to_csv(self, save_dir_path: str) -> None:
        if isinstance(self.results_df, pd.DataFrame):
            self.results_df.to_csv(os.path.join(save_dir_path, "predict_result.csv"))
        else:
            logger.warning("Results dataframe is not initialized")

    def save_metrics_df_to_csv(self, save_dir_path: str) -> None:
        self.metrics_df.to_csv(os.path.join(save_dir_path, "predict_metrics.csv"))

    def scatter_plot(self, save_dir_path: str) -> None:
        """This function creates scatter plot of prediction and observation data of `results_df`."""
        output_param_name = self.output_parameter_names[0]
        target_time_steps = [0, 1, 2]

        all_cases_scatter_plot(
            result_df=self.query_result_df(),
            downstream_directory=save_dir_path,
            output_param_name=output_param_name,
            r2_score=self.r2_score_from_results_df(output_param_name=output_param_name),
        )

        # Only use first 3 step data
        all_cases_scatter_plot(
            result_df=self.query_result_df(target_time_steps=target_time_steps),
            downstream_directory=save_dir_path,
            output_param_name=output_param_name,
            r2_score=self.r2_score_from_results_df(output_param_name, target_time_steps=target_time_steps),
            save_fig_name="first-3step-all-cases.png",
        )

        unique_dates = self.results_df["date"].unique().tolist()
        for date in unique_dates:
            date_scatter_plot(
                result_df=self.query_result_df(target_date=date),
                downstream_directory=save_dir_path,
                output_param_name=output_param_name,
                date=date,
                r2_score=self.r2_score_from_results_df(output_param_name=output_param_name, target_date=date),
            )

            # Only use first 3 step data.
            date_scatter_plot(
                result_df=self.query_result_df(target_date=date, target_time_steps=target_time_steps),
                downstream_directory=save_dir_path,
                output_param_name=output_param_name,
                date=date,
                r2_score=self.r2_score_from_results_df(
                    output_param_name=output_param_name, target_date=date, target_time_steps=target_time_steps
                ),
                save_fig_name=f"first-3step-{date}-cases.png",
            )

        # Casetype
        tc_case_result_df = self.query_result_df(is_tc_case=True)
        if not tc_case_result_df.empty:
            casetype_scatter_plot(
                result_df=tc_case_result_df,
                case_type="tc",
                downstream_directory=save_dir_path,
                output_param_name=output_param_name,
                r2_score=self.r2_score_from_results_df(output_param_name=output_param_name, is_tc_case=True),
            )

        not_tc_case_result_df = self.query_result_df(is_tc_case=True)
        if not not_tc_case_result_df.empty:
            casetype_scatter_plot(
                result_df=not_tc_case_result_df,
                case_type="not_tc",
                downstream_directory=save_dir_path,
                output_param_name=output_param_name,
                r2_score=self.r2_score_from_results_df(output_param_name=output_param_name, is_tc_case=False),
            )

    def geo_plot(self, test_case_name: str, save_dir_path: str, pred_tensors: torch.Tensor) -> None:
        """This function create and save geo plotted images (Mainly used for rainfall images).
            And saving prediction grid array.

        Args:
            test_case_name (str): A test case name of the prediction.
            save_dir_path (str): Save directory path.
            pred_tensors (torch.Tensor): Prediction tensors of a test case.
                The shape is [batch=1, n_featuures=1, seq_length, height, width]. This tensor should be scaled.
        """
        for time_step in range(self.hydra_cfg.label_seq_length):
            pred_ndarray = pred_tensors[0, 0, time_step, ...].cpu().detach().numpy().copy()
            utc_time_name = self.get_prediction_utc_time(test_case_name, time_step)
            if self.hydra_cfg.use_dummy_data is False:
                geoimg_interactor = GeoimgGenratorInteractor()
                geoimg_interactor.save_img(
                    self.output_parameter_names[0],
                    pred_ndarray,
                    self.observation_point_file_path,
                    os.path.join(save_dir_path, f"{utc_time_name}.png"),
                )
            save_parquet(
                pred_ndarray,
                os.path.join(save_dir_path, f"{utc_time_name}.parquet.gzip"),
                self.observation_point_file_path,
            )

    def update_input_tensor(
        self, before_input_tensor: torch.Tensor, before_standarized_info: Dict, next_frame_tensor: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Update input tensor X_test for next prediction. A next one frame tensor (prediction or label tensor) a given
        and update the first frame of X_test with that given tensor.

        Args:
            before_input_tensor (torch.Tensor):
            before_standarized_info (Dict): Stadarization information to rescale tensor using this.
            next_frame_tensor (torch.Tensor): This tensor should be scaled to [0, 1].
        """
        if next_frame_tensor.max().item() > 1 or next_frame_tensor.min().item() < 0:
            raise ValueError(
                "next_frame_tensor is not scaled to [0, 1], "
                f"but [{next_frame_tensor.min().item(), next_frame_tensor.max().item()}]"
            )

        # Convert next_frame tensor to grids if ob point tensor is given.
        _, num_channels, _, height, width = before_input_tensor.size()
        if next_frame_tensor.ndim == 2:
            # The case of next_frame_tensor is [ob_point values]
            _next_frame_tensor = next_frame_tensor.cpu().detach()
            _next_frame_tensor = normalize_tensor(_next_frame_tensor, device="cpu")
            _next_frame_ndarray = _next_frame_tensor.numpy().copy()
            next_frame_tensor = torch.zeros(
                (len(self.input_parameter_names), width, height), dtype=torch.float, device=DEVICE
            )
            for param_dim, weather_param in enumerate(self.input_parameter_names):
                interpolator_interactor = InterpolatorInteractor()
                interp_next_frame_ndarray = interpolator_interactor.interpolate(
                    weather_param, _next_frame_ndarray[param_dim, ...], self.observation_point_file_path
                )
                next_frame_tensor[param_dim, ...] = torch.from_numpy(interp_next_frame_ndarray.copy()).to(DEVICE)
            next_frame_tensor = normalize_tensor(next_frame_tensor, device=DEVICE)

        scaling_method = self.hydra_cfg.scaling_method

        if scaling_method == ScalingMethod.MinMax.value:
            return (
                torch.cat(
                    (
                        before_input_tensor[:, :, 1:, ...],
                        torch.reshape(next_frame_tensor, (1, num_channels, 1, height, width)),
                    ),
                    dim=2,
                ),
                {},
            )

        # elif scaling_method == ScalingMethod.Standard.value or scaling_method == ScalingMethod.MinMaxStandard.value:
        else:
            for param_dim, param_name in enumerate(self.input_parameter_names):
                means, stds = before_standarized_info[param_name]["mean"], before_standarized_info[param_name]["std"]
                before_input_tensor[:, param_dim, ...] = before_input_tensor[:, param_dim, ...] * stds + means

            updated_input_tensor = torch.cat(
                (
                    before_input_tensor[:, :, 1:, ...],
                    torch.reshape(next_frame_tensor, (1, num_channels, 1, *before_input_tensor.size()[3:])),
                ),
                dim=2,
            )
            standarized_info = {}
            for param_dim, param_name in enumerate(self.input_parameter_names):
                standarized_info[param_name] = {}
                means = torch.mean(updated_input_tensor[:, param_dim, ...])
                stds = torch.std(updated_input_tensor[:, param_dim, ...])
                updated_input_tensor[:, param_dim, ...] = (updated_input_tensor[:, param_dim, ...] - means) / stds
                standarized_info[param_name]["mean"] = means
                standarized_info[param_name]["std"] = stds
            return updated_input_tensor, standarized_info

    def get_timeseries_metrics_df(self, target_param_name: str) -> pd.DataFrame:
        """
        This function create timeseries metrics (rmse, r2_score) from results_df.
        Each metrics are calculated by grouping all test cases together by time step.

        Args:
            target_param_name(str): weather parameter names like rain, temperature e.t.c
        Return:
            pd.DataFrame: columns are time_step, test_case_name, rmse
        """
        target_poteka_col = PPOTEKACols.get_col_from_weather_param(target_param_name)
        df = pd.DataFrame(columns=["time_step", "test_case_name", "rmse", "r2_score"])
        for time_step, time_step_df in self.results_df.groupby("time_step"):
            for test_case_name, test_case_result_df in time_step_df.groupby("test_case_name"):
                rmse = self.calc_rmse(
                    test_case_result_df[target_poteka_col].astype(float).to_numpy(),
                    test_case_result_df["Pred_Value"].astype(float).to_numpy(),
                )
                r2_score_val = self.calc_r2_score(
                    test_case_result_df[target_poteka_col].astype(float).to_numpy(),
                    test_case_result_df["Pred_Value"].astype(float).to_numpy(),
                )
                df = pd.concat(
                    [
                        df,
                        pd.DataFrame(
                            [
                                {
                                    "time_step": time_step,
                                    "test_case_name": test_case_name,
                                    "rmse": rmse,
                                    "r2_score": r2_score_val,
                                }
                            ]
                        ),
                    ]
                )

        df["time_step"] = df["time_step"].astype(int)
        df["rmse"] = df["rmse"].astype(float)
        df["r2_score"] = df["r2_score"].astype(float)

        return df

    def timeseries_metrics_boxplot(
        self, target_param_name: str, target_metrics_name: str, downstream_directory: str
    ) -> None:
        """
        This function create timeseries box plot of metrics(rmse, r2_score) from result_df.

        Args:
            target_param_name(str): Weather parameter name like rain, tenperature ...
            target_metrics_name(str): rmse or r2_score
            downstream_directory(str): The directory path to save the plto figure.
        """
        if target_metrics_name not in ["rmse", "r2_score"]:
            raise ValueError(f"Invalid metrics name: {target_metrics_name}.")

        plot_df = self.get_timeseries_metrics_df(target_param_name)
        plt.figure(figsize=(8, 6))
        ax = sns.boxplot(data=plot_df, x="time_step", y=target_metrics_name)
        ax.set_title(f"{target_metrics_name} timeseries change for prediction of {target_param_name}.")
        ax.set_xlabel("prediction time step (min)")
        ax.set_ylabel(f"{target_metrics_name}")
        time_step = self.hydra_cfg.preprocess.time_step_minutes
        ax.set_xticklabels([i * time_step + time_step for i in range(self.hydra_cfg.label_seq_length)])
        plt.savefig(os.path.join(downstream_directory, f"timeseries_{target_metrics_name}_plot.png"))
        plt.close()

    def save_attention_maps(self, save_dir_path: str) -> None:
        get_attention_maps = getattr(self.model, "get_attention_maps", None)
        if not callable(get_attention_maps):
            raise ValueError(f"{self.model.__class__.__name__} does not have `get_attention_maps` method.")

        geoimg_interactor = GeoimgGenratorInteractor()

        # NOTE: attention maps shape is (batch_size=1, seq_len, height * width)
        # Extracted only center attention map because of memory usage.
        for (layer_name, attention_maps) in get_attention_maps().items():
            layer_name = layer_name.split(".")[-1]
            for seq_idx in range(attention_maps.size(1)):
                # save only attention maps of center
                target_maps = attention_maps[0, seq_idx]
                target_maps = (
                    torch.reshape(target_maps, (GridSize.HEIGHT, GridSize.WIDTH)).cpu().detach().numpy().copy()
                )

                save_dir = os.path.join(save_dir_path, "attention_maps", layer_name)
                os.makedirs(save_dir, exist_ok=True)
                # Save as npy file.
                with open(os.path.join(save_dir, f"attentionMap-seq{seq_idx}.npy"), "wb") as fp:
                    np.save(fp, target_maps)

                # Save as Image
                geoimg_interactor.save_img(
                    "attention_map",
                    target_maps,
                    self.observation_point_file_path,
                    os.path.join(save_dir, f"attentionMap-sequence{seq_idx}.png"),
                )
