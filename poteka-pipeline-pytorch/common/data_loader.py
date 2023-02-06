import logging
from typing import List, Optional, Tuple, Dict, OrderedDict
import json
from collections import OrderedDict as ordered_dict
import os

from tqdm import tqdm
import pandas as pd
import numpy as np
import torch

from common.utils import calc_u_v, get_ob_point_values_from_tensor, load_scaled_data
from common.custom_logger import CustomLogger
from common.config import GridSize, MinMaxScalingValue, PPOTEKACols, ScalingMethod

logger = CustomLogger("data_loader_Logger", level=logging.DEBUG)


def train_data_loader(
    meta_data_file_path: str,
    observation_point_file_path: str,
    isMaxSizeLimit: bool = False,
    scaling_method: str = "min_max",
    isObPointLabelData: bool = False,
    debug_mode: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Load two tensors (input and label) and return them.

    This function load the datasets of meta_data_file_path and scaling them.

    Args:
        meta_data_file_path (str): The information of the datafile paths.
            The file is automatically generated in `preprocess` step.
        observation_point_file_path (str): The information of the PPOTEKA observation points.
        isMaxSizeLimit (bool): If true, the max length of the dataset is 100.
        scaling_method (str): The scaling method.
        isObPointLabelData (bool): If true, the label data is observation point shape like (35,).

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: (input tensor, label tensor)

    """
    if not ScalingMethod.is_valid(scaling_method):
        raise ValueError("Invalid scaling method")
    # [TODO]
    # You may add these args to data_laoder()?
    # HEIGHT, WIDTH = 50, 50
    HEIGHT, WIDTH = GridSize.HEIGHT, GridSize.WIDTH
    meta_file = json_loader(meta_data_file_path)
    meta_file_paths = meta_file["file_paths"]
    # =============================
    # meta_file_paths: List[Dict]
    # [ param1: {
    #   input: [6 paths],
    #   label: [1 paths],
    #   },
    #   param2: {
    #       ...
    #   }, ...
    # }]
    with open(observation_point_file_path, "r") as f:
        ob_point_data = json.load(f)
    ob_point_count = len(list(ob_point_data.keys()))
    logger.info(f"Scaling method: {scaling_method}")
    num_channels = len(meta_file_paths[0].keys())
    input_seq_length = len(meta_file_paths[0]["rain"]["input"])
    label_seq_length = len(meta_file_paths[0]["rain"]["label"])
    meta_file_paths = meta_file_paths[:100] if isMaxSizeLimit else meta_file_paths
    # [TODO]
    # Tensor shape should be (batch_size, num_channels, seq_len, height, width)
    input_tensor = torch.zeros((len(meta_file_paths), num_channels, input_seq_length, HEIGHT, WIDTH), dtype=torch.float)

    if isObPointLabelData is True:
        label_tensor = torch.zeros(
            (len(meta_file_paths), num_channels, label_seq_length, ob_point_count), dtype=torch.float
        )
    else:
        label_tensor = torch.zeros(
            (len(meta_file_paths), num_channels, label_seq_length, HEIGHT, WIDTH), dtype=torch.float
        )

    for dataset_idx, dataset_path in tqdm(
        enumerate(meta_file_paths), ascii=True, desc="Loading Train and Valid dataset"
    ):
        # load input data
        # input data is scaled in 2 ways
        # 1. MinMax: scaled to [0, 1]
        # 2. MinMaxStandard: scaleed to [0, 1] first, then scaled with standarization
        for param_idx, param_name in enumerate(dataset_path.keys()):
            # store input data
            _, _ = store_input_data(
                dataset_idx=dataset_idx,
                param_idx=param_idx,
                input_tensor=input_tensor,
                input_dataset_paths=dataset_path[param_name]["input"],
                scaling_method=scaling_method,
                inplace=True,
            )
            # load label data
            if isObPointLabelData is True:
                _store_label_data(
                    observation_point_file_path=observation_point_file_path,
                    dataset_idx=dataset_idx,
                    param_idx=param_idx,
                    label_tensor=label_tensor,
                    label_dataset_paths=dataset_path[param_name]["label"],
                    inplace=True,
                )
            else:
                store_label_data(
                    dataset_idx=dataset_idx,
                    param_idx=param_idx,
                    label_tensor=label_tensor,
                    label_dataset_paths=dataset_path[param_name]["label"],
                    inplace=True,
                )
    logger.info(f"Input tensor shape: {input_tensor.shape}")
    logger.info(f"Label tensor shape: {label_tensor.shape}")
    return (input_tensor, label_tensor)


def test_data_loader(
    meta_data_file_path: str,
    observation_point_file_path: str,
    scaling_method: str = "min_max",
    use_dummy_data: bool = False,
    isObPointLabelData: bool = False,
) -> Tuple[Dict, OrderedDict]:
    """Load the input and label dataset of each test cases.

    Return the input and label dataset with the test case infomation.

    Args:
        meta_data_file_path (str): The information of the datafile paths.
            The file is automatically generated in `preprocess` step.
        observation_point_file_path (str): The information of the PPOTEKA observation points.
        scaling_method (str): The scaling method.
        use_dummy_data (bool): If true, the dummy dataset is generated and returned.
        isObPointLabelData (bool): If true, the label data is observation point shape like (35,).

    Returns:
        Tuple[Dict, OrderedDict]: (
            input & label tensor and other informations,
            parameter names with the channel index.
    )

    """
    if not ScalingMethod.is_valid(scaling_method):
        raise ValueError("Invalid scaling method")
    # [TODO]
    # You may add these args to data_laoder()?
    # HEIGHT, WIDTH = 50, 50
    HEIGHT, WIDTH = GridSize.HEIGHT, GridSize.WIDTH
    meta_file = json_loader(meta_data_file_path)
    meta_file_paths = meta_file["file_paths"]
    # =============================
    # meta_file_paths: Dict
    # { sample1: {
    #     date: ###,
    #     start: ###,
    #     rain: {
    #       input: [6 paths],
    #       label: [6 paths],
    #     },
    #     humidity: { input: [...]},
    #     temperature: { input: [...]},
    #     ...
    #   },
    #   sample2: {...}
    # }]
    with open(observation_point_file_path, "r") as f:
        ob_point_data = json.load(f)
    ob_point_names = list(ob_point_data.keys())
    ob_point_count = len(ob_point_names)
    logger.info(f"Scaling method: {scaling_method}")
    output_data = {}
    features_dict = ordered_dict()
    for sample_name in tqdm(meta_file_paths.keys(), ascii=True, desc="Loading Valid dataset"):
        feature_names = [v for v in meta_file_paths[sample_name].keys() if v not in ["date", "start"]]
        if bool(features_dict) is False:
            for idx, name in enumerate(feature_names):
                features_dict[idx] = name
        num_channels = len(feature_names)
        input_seq_length = len(meta_file_paths[sample_name]["rain"]["input"])
        label_seq_length = len(meta_file_paths[sample_name]["rain"]["label"])
        input_tensor = torch.zeros((1, num_channels, input_seq_length, HEIGHT, WIDTH), dtype=torch.float)

        if isObPointLabelData is True:
            label_tensor = torch.zeros(
                (len(meta_file_paths), num_channels, label_seq_length, ob_point_count), dtype=torch.float
            )
        else:
            label_tensor = torch.zeros(
                (len(meta_file_paths), num_channels, label_seq_length, HEIGHT, WIDTH), dtype=torch.float
            )

        standarize_info = {}
        # load input data
        # input data is scaled in 2 ways
        # 1. MinMax: scaled to [0, 1]
        # 2. MinMaxStandard: scaleed to [0, 1] first, then scaled with standarization
        for param_idx, param_name in enumerate(feature_names):
            standarized_info, _ = store_input_data(
                dataset_idx=0,
                param_idx=param_idx,
                input_tensor=input_tensor,
                input_dataset_paths=meta_file_paths[sample_name][param_name]["input"],
                scaling_method=scaling_method,
                inplace=True,
            )
            standarize_info[param_name] = standarized_info
            # load label data
            # label data is scaled to [0, 1]
            if isObPointLabelData is True:
                _store_label_data(
                    observation_point_file_path=observation_point_file_path,
                    dataset_idx=0,
                    param_idx=param_idx,
                    label_tensor=label_tensor,
                    label_dataset_paths=meta_file_paths[sample_name][param_name]["label"],
                    inplace=True,
                )
            else:
                store_label_data(
                    dataset_idx=0,
                    param_idx=param_idx,
                    label_tensor=label_tensor,
                    label_dataset_paths=meta_file_paths[sample_name][param_name]["label"],
                    inplace=True,
                )
        # Load One Day data for evaluation
        label_dfs = {}
        if use_dummy_data:
            for i in range(label_seq_length):
                data = {}
                for col in PPOTEKACols.get_cols():
                    min_val, max_val = MinMaxScalingValue.get_minmax_values_by_ppoteka_cols(col)
                    data[col] = np.random.uniform(low=min_val, high=max_val, size=(ob_point_count))
                label_dfs[i] = pd.DataFrame(data, index=ob_point_names)

        else:
            # If you use dummy data, parqet files of one_data_data don't exist.
            for i in range(label_seq_length):
                df_path = meta_file_paths[sample_name]["rain"]["label"][i]
                df_path = df_path.replace("rain_image", "one_day_data")  # ~.csv
                if os.path.exists(df_path):
                    df = pd.read_csv(df_path, index_col=0)
                elif os.path.exists(df_path.replace(".csv", ".parquet.gzip")):
                    df = pd.read_parquet(df_path.replace(".csv", ".parquet.gzip"), engine="pyarrow")
                    df.set_index("Unnamed: 0", inplace=True)
                else:
                    raise ValueError(f"the file does not exist {df_path} (.parquet.gzip)")
                # calculate u, v wind
                uv_wind_df = pd.DataFrame(
                    [calc_u_v(df.loc[i, :], i) for i in df.index],
                    columns=["OB_POINT", PPOTEKACols.U_WIND.value, PPOTEKACols.V_WIND.value],
                )
                uv_wind_df.set_index("OB_POINT", inplace=True)
                df = df.merge(uv_wind_df, left_index=True, right_index=True)
                label_dfs[i] = df

        output_data[sample_name] = {
            "date": meta_file_paths[sample_name]["date"],
            "start": meta_file_paths[sample_name]["start"],
            "input": input_tensor,
            "label": label_tensor,
            "label_df": label_dfs,
            "standarize_info": standarize_info,
        }

    return output_data, features_dict


def store_input_data(
    dataset_idx: int,
    param_idx: int,
    input_tensor: torch.Tensor,
    input_dataset_paths: List[str],
    scaling_method: str,
    inplace: bool = False,
) -> Tuple[Dict[str, float], Optional[torch.Tensor]]:
    """Load a single batch data and store it to the input tensors.

    Args:
        dataset_idx (int): The batch index.
        param_idx (int): The channel index
        input_tensor (torch.Tensor): The base input_tensor (Batch, Sequences, Channels, Height, Width).
        input_dataset_paths (List[str]): The data file paths of a certain paramter and time.
        scaling_method (str): The scaling method.
        inplace (bool): If false, the stored input data is returned.

    Returns:
        Tuple[Dict[str, float], Optional[torch.Tensor]]: (
            standarized_info,
            input_tensor,
        )
        `standarized_info` is used for re-standarized.
    """
    for seq_idx, data_file_path in enumerate(input_dataset_paths):
        numpy_arr = load_scaled_data(data_file_path)

        if np.isnan(numpy_arr).any():
            logger.error(f"NaN value contains in {data_file_path}")

        input_tensor[dataset_idx, param_idx, seq_idx, :, :] = torch.from_numpy(numpy_arr)

    standarized_info = {"mean": 0, "std": 1.0}
    if scaling_method == ScalingMethod.Standard.value or scaling_method == ScalingMethod.MinMaxStandard.value:
        means = torch.mean(input_tensor[dataset_idx, param_idx, :, :, :])
        stds = torch.std(input_tensor[dataset_idx, param_idx, :, :, :])
        input_tensor[dataset_idx, param_idx, :, :, :] = (input_tensor[dataset_idx, param_idx, :, :, :] - means) / stds
        standarized_info["mean"] = float(means)
        standarized_info["std"] = float(stds)

    if not inplace:
        return standarized_info, input_tensor

    return standarized_info, None


def store_label_data(
    dataset_idx: int,
    param_idx: int,
    label_tensor: torch.Tensor,
    label_dataset_paths: List[str],
    inplace: bool = False,
) -> Optional[torch.Tensor]:
    """Load a single batch data and store it to the label tensor.

    Args:
        dataset_idx (int): The batch index.
        param_idx (int): The channel index
        input_tensor (torch.Tensor): The base input_tensor (Batch, Sequences, Channels, Height, Width).
        input_dataset_paths (List[str]): The data file paths of a certain paramter and time.
        scaling_method (str): The scaling method.
        inplace (bool): If false, the stored input data is returned.

    Returns:
        Tuple[Dict[str, float], Optional[torch.Tensor]]: (
            standarized_info,
            input_tensor,
        )
        `standarized_info` is used for re-standarized.
    """
    for seq_idx, data_file_path in enumerate(label_dataset_paths):
        numpy_arr = load_scaled_data(data_file_path)

        if np.isnan(numpy_arr).any():
            logger.error(f"NaN value contains in {data_file_path}")

        label_tensor[dataset_idx, param_idx, seq_idx, :, :] = torch.from_numpy(numpy_arr)

    if not inplace:
        return label_tensor


def _store_label_data(
    observation_point_file_path: str,
    dataset_idx: int,
    param_idx: int,
    label_tensor: torch.Tensor,
    label_dataset_paths: List[str],
    inplace: bool = True,
) -> Optional[torch.Tensor]:
    """Load a single batch data and store it to the label tensor.

    Before storeing, the observation point data are extracting from a grid data.

    Args:
        observation_point_file_path (str): The information of the PPOTEKA observation points.
        dataset_idx (int): The batch index.
        param_idx (int): The channel index
        label_tensor (torch.Tensor): The base label tensor (Batch=1, Sequences, Channels, Height, Width).
        label_dataset_paths (List[str]): The list of the file paths of label dataset.
        inplace (bool): If false, the stored label tensor data is returned.

    Returns:
        Optional[torch.Tensor]: If `inplace=True`, the stored label tensor is returned.
    """
    for seq_idx, data_file_path in enumerate(label_dataset_paths):
        numpy_arr = load_scaled_data(data_file_path)  # The array shape is (50, 50)

        if np.isnan(numpy_arr).any():
            logger.error(f"NaN value contains in {data_file_path}")

        data_tensor = torch.from_numpy(numpy_arr)
        label_tensor[dataset_idx, param_idx, seq_idx, :] = get_ob_point_values_from_tensor(
            data_tensor, observation_point_file_path
        )  # The output shape is [the number of observation point]
    if not inplace:
        return label_tensor


def json_loader(path: str):
    """Load json file"""
    with open(path, "r") as f:
        data = json.load(f)
    return data


def sample_data_loader(
    train_size: int,
    valid_size: int,
    x_batch: int,
    y_batch: int,
    height: int,
    width: int,
    vector_size: int,
):
    """Generate dummy datasets."""
    X_train = random_normalized_data(train_size, x_batch, height, width, vector_size)
    y_train = random_normalized_data(train_size, y_batch, height, width, vector_size)
    X_valid = random_normalized_data(valid_size, x_batch, height, width, vector_size)
    y_valid = random_normalized_data(valid_size, y_batch, height, width, vector_size)

    return (X_train, y_train), (X_valid, y_valid)


def random_normalized_data(
    sample_size: int,
    batch_num: int,
    height: int,
    width: int,
    vector_size: int,
):
    """Generate normalized data"""
    arr = np.array([[np.random.rand(height, width, vector_size)] * batch_num] * sample_size)
    return arr
