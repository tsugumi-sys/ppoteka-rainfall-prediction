import logging
import os
import sys
from typing import Dict, List

import pandas as pd
from numpy.random import sample

sys.path.append("..")
from common.config import WEATHER_PARAMS  # noqa: E402
from common.utils import param_date_path, timestep_csv_names  # noqa: E402

logger = logging.getLogger(__name__)


def get_train_data_files(
    project_root_dir_path: str,
    train_list_df: pd.DataFrame,
    input_parameters: List[str] = ["rain", "temperature"],
    time_step_minutes: int = 10,
    time_slides_delta: int = 3,
    input_seq_length: int = 6,
    label_seq_length: int = 6,
) -> List[Dict]:
    """Get train data file paths.

    Args:
        train_list_df (pd.DataFrame): pandas.DataFrame with training data informations
            with columns ['date', 'start_time', 'end_time']
        input_parameters (List[str], optional): Input parameters list. Defaults to ["rain", "temperature"].
        time_step_minutes (int, optional): Time step of datesets in minutes. Defaults to 10.
        time_slides_delta (int, optional): Time slides used for creating datesets in minutes. Defaults to 3.

    Raises:
        ValueError: `rain` must be in `input_parameters`.
        ValueError: check if all input parameters in `input_paramerers` are valid.

    Returns:
        List[Dict]: list of dictioaryis contains data file paths of each input tarameters.
            {
                "rain": {"input": ['path/to/datafiles/0-0.csv', 'path/to/datafiles/0-10.csv', ...], "label": [...]},
                "temperature": {"input": [...], "label": [...]},
                ...
            }
    """
    if WEATHER_PARAMS.RAIN.value not in input_parameters:
        logger.error(f"rain is not in {input_parameters}")
        raise ValueError("input_parameters should have 'rain'.")

    if not WEATHER_PARAMS.is_params_valid(input_parameters):
        logger.error(f"{input_parameters} is invalid name.")
        raise ValueError(f"preprocess_input_parameters should be in {WEATHER_PARAMS.valid_params()}")

    _timestep_csv_names = timestep_csv_names(time_step_minutes=time_step_minutes)
    paths = []
    for idx in train_list_df.index:
        date = train_list_df.loc[idx, "date"]
        year, month = date.split("-")[0], date.split("-")[1]

        input_parameters_date_paths = {}
        if len(input_parameters) > 0:
            for pa in input_parameters:
                input_parameters_date_paths[pa] = os.path.join(
                    project_root_dir_path,
                    param_date_path(pa, year, month, date),
                )

        start, end = train_list_df.loc[idx, "start_time"], train_list_df.loc[idx, "end_time"]
        idx_start, idx_end = _timestep_csv_names.index(str(start) + ".csv"), _timestep_csv_names.index(
            str(end) + ".csv"
        )
        idx_start = idx_start - label_seq_length if idx_start > label_seq_length else 0
        idx_end = (
            idx_end + label_seq_length
            if idx_end < len(_timestep_csv_names) - label_seq_length
            else len(_timestep_csv_names) - 1
        )
        for i in range(idx_start, idx_end - label_seq_length, time_slides_delta):
            h_m_csv_names = _timestep_csv_names[i : i + input_seq_length + label_seq_length]  # noqa: E226,E203

            _tmp = {}
            for pa in input_parameters:
                if pa == WEATHER_PARAMS.WIND.value:
                    for name in [WEATHER_PARAMS.U_WIND.value, WEATHER_PARAMS.V_WIND.value]:
                        _tmp[name] = {
                            "input": [],
                            "label": [],
                        }
                else:
                    _tmp[pa] = {
                        "input": [],
                        "label": [],
                    }
            # load input data
            for input_h_m_csv_name in h_m_csv_names[:input_seq_length]:
                for pa in input_parameters:
                    if pa == WEATHER_PARAMS.WIND.value:
                        _tmp[WEATHER_PARAMS.U_WIND.value]["input"] += [
                            input_parameters_date_paths[pa] + f"/{input_h_m_csv_name}".replace(".csv", "U.csv")
                        ]
                        _tmp[WEATHER_PARAMS.V_WIND.value]["input"] += [
                            input_parameters_date_paths[pa] + f"/{input_h_m_csv_name}".replace(".csv", "V.csv")
                        ]
                    elif pa == WEATHER_PARAMS.U_WIND.value:
                        _tmp[WEATHER_PARAMS.U_WIND.value]["input"] += [
                            input_parameters_date_paths[pa] + f"/{input_h_m_csv_name}".replace(".csv", "U.csv")
                        ]
                    elif pa == WEATHER_PARAMS.V_WIND.value:
                        _tmp[WEATHER_PARAMS.V_WIND.value]["input"] += [
                            input_parameters_date_paths[pa] + f"/{input_h_m_csv_name}".replace(".csv", "V.csv")
                        ]
                    else:
                        _tmp[pa]["input"] += [input_parameters_date_paths[pa] + f"/{input_h_m_csv_name}"]
            # load label data
            for label_h_m_csv_name in h_m_csv_names[input_seq_length:]:
                for pa in input_parameters:
                    if pa == WEATHER_PARAMS.WIND.value:
                        _tmp[WEATHER_PARAMS.U_WIND.value]["label"] += [
                            input_parameters_date_paths[pa] + f"/{label_h_m_csv_name}".replace(".csv", "U.csv")
                        ]
                        _tmp[WEATHER_PARAMS.V_WIND.value]["label"] += [
                            input_parameters_date_paths[pa] + f"/{label_h_m_csv_name}".replace(".csv", "V.csv")
                        ]
                    elif pa == WEATHER_PARAMS.U_WIND.value:
                        _tmp[WEATHER_PARAMS.U_WIND.value]["label"] += [
                            input_parameters_date_paths[pa] + f"/{label_h_m_csv_name}".replace(".csv", "U.csv")
                        ]
                    elif pa == WEATHER_PARAMS.V_WIND.value:
                        _tmp[WEATHER_PARAMS.V_WIND.value]["label"] += [
                            input_parameters_date_paths[pa] + f"/{label_h_m_csv_name}".replace(".csv", "V.csv")
                        ]
                    else:
                        _tmp[pa]["label"] += [input_parameters_date_paths[pa] + f"/{label_h_m_csv_name}"]

            paths.append(_tmp)
    return paths


def get_test_data_files(
    project_root_dir_path: str,
    test_data_list: Dict,
    input_parameters: List[str] = ["rain", "temperature"],
    time_step_minutes: int = 10,
    input_seq_length: int = 6,
    label_seq_length: int = 6,
) -> Dict:
    """Get test data file informations

    Args:
        test_data_list (Dict): test data information contains date, start, end.
        input_parameters (List[str], optional): Input parameters list. Defaults to ["rain", "temperature"].
        time_step_minutes (int, optional): time step minutes. Defaults to 10.

    Raises:
        ValueError: when rain is not in input_parameters
        ValueError: when invalid parameter name contains

    Returns:
        Dict: data file paths of each test cases like following.
            {
                "case1": {
                    "rain": {"input": ['path/to/datafiles/0-0.csv', 'path/to/datafiles/0-10.csv', ...], "label": [...]},
                    "temperature": {"input": [...], "label": [...]},
                    ...,
                    "date": "2020/01/05",
                    "start": "1000UTC",
                },
                "case2": {...}
            }
    """
    if WEATHER_PARAMS.RAIN.value not in input_parameters:
        logger.error(f"rain is not in {input_parameters}")
        raise ValueError("preprocess_input_parameters should have 'rain'.")

    if not WEATHER_PARAMS.is_params_valid(input_parameters):
        logger.error(f"{input_parameters} is invalid name.")
        raise ValueError(f"preprocess_input_parameters should be in {WEATHER_PARAMS.valid_params()}")

    _timestep_csv_names = timestep_csv_names(time_step_minutes=time_step_minutes)
    paths = {}
    for case_name in test_data_list.keys():
        for sample_name in test_data_list[case_name].keys():
            for idx in test_data_list[case_name][sample_name].keys():
                if int(idx) > len(_timestep_csv_names) - 1:
                    logger.error(
                        f"case name: {case_name} - sample name: {sample_name}({idx}): The label_seq_length is too big"
                    )
                    break
                sample_info = test_data_list[case_name][sample_name][idx]
                date = sample_info["date"]
                year, month = date.split("-")[0], date.split("-")[1]

                input_parameters_date_paths = {}
                if len(input_parameters) > 0:
                    for pa in input_parameters:
                        input_parameters_date_paths[pa] = os.path.join(
                            project_root_dir_path,
                            param_date_path(pa, year, month, date),
                        )

                # NOTE: `start` means the start of prediction.
                # If start is 5:00, inputs are before 5:00. label should be after 5:00.
                start = sample_info["start"]
                pred_start_idx = _timestep_csv_names.index(str(start))
                input_start_idx = pred_start_idx - input_seq_length
                label_end_idx = pred_start_idx + label_seq_length

                if input_start_idx < 0:
                    logger.error(f"input data is not enough. Skip case_name: {case_name} - sample_name: {sample}")
                    break

                if label_end_idx > len(_timestep_csv_names) - 1:
                    logger.error(f"label data is not enough. Skip case_name: {case_name} - sample_name: {sample_name}")
                    break

                _tmp = {}
                for pa in input_parameters:
                    if pa == WEATHER_PARAMS.WIND.value:
                        for name in [WEATHER_PARAMS.U_WIND.value, WEATHER_PARAMS.V_WIND.value]:
                            _tmp[name] = {
                                "input": [],
                                "label": [],
                            }
                    else:
                        _tmp[pa] = {
                            "input": [],
                            "label": [],
                        }

                # Load input data
                for input_h_m_csv_name in _timestep_csv_names[input_start_idx:pred_start_idx]:
                    for pa in input_parameters:
                        if pa == WEATHER_PARAMS.WIND.value:
                            _tmp[WEATHER_PARAMS.U_WIND.value]["input"] += [
                                input_parameters_date_paths[pa] + f"/{input_h_m_csv_name}".replace(".csv", "U.csv")
                            ]
                            _tmp[WEATHER_PARAMS.V_WIND.value]["input"] += [
                                input_parameters_date_paths[pa] + f"/{input_h_m_csv_name}".replace(".csv", "V.csv")
                            ]
                        elif pa == WEATHER_PARAMS.U_WIND.value:
                            _tmp[WEATHER_PARAMS.U_WIND.value]["input"] += [
                                input_parameters_date_paths[pa] + f"/{input_h_m_csv_name}".replace(".csv", "U.csv")
                            ]
                        elif pa == WEATHER_PARAMS.V_WIND.value:
                            _tmp[WEATHER_PARAMS.V_WIND.value]["input"] += [
                                input_parameters_date_paths[pa] + f"/{input_h_m_csv_name}".replace(".csv", "V.csv")
                            ]
                        else:
                            _tmp[pa]["input"] += [input_parameters_date_paths[pa] + f"/{input_h_m_csv_name}"]
                # Load label data
                # contains other parameters value for sequential prediction
                for label_h_m_csv_name in _timestep_csv_names[pred_start_idx:label_end_idx]:
                    for pa in input_parameters:
                        if pa == WEATHER_PARAMS.WIND.value:
                            _tmp[WEATHER_PARAMS.U_WIND.value]["label"] += [
                                input_parameters_date_paths[pa] + f"/{label_h_m_csv_name}".replace(".csv", "U.csv")
                            ]
                            _tmp[WEATHER_PARAMS.V_WIND.value]["label"] += [
                                input_parameters_date_paths[pa] + f"/{label_h_m_csv_name}".replace(".csv", "V.csv")
                            ]
                        elif pa == WEATHER_PARAMS.U_WIND.value:
                            _tmp[WEATHER_PARAMS.U_WIND.value]["label"] += [
                                input_parameters_date_paths[pa] + f"/{label_h_m_csv_name}".replace(".csv", "U.csv")
                            ]
                        elif pa == WEATHER_PARAMS.V_WIND.value:
                            _tmp[WEATHER_PARAMS.V_WIND.value]["label"] += [
                                input_parameters_date_paths[pa] + f"/{label_h_m_csv_name}".replace(".csv", "V.csv")
                            ]
                        else:
                            _tmp[pa]["label"] += [input_parameters_date_paths[pa] + f"/{label_h_m_csv_name}"]

                _sample_name = f"{case_name}_{date}_{start.replace('.csv', '')}_start"
                paths[_sample_name] = _tmp
                paths[_sample_name]["date"] = date
                paths[_sample_name]["start"] = start
    return paths
