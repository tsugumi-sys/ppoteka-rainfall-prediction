import argparse
import sys
import itertools
import json
import os
import re
from pathlib import Path
from typing import Dict, List

import pandas as pd
from pydantic import BaseModel

sys.path.append(".")
from common.utils import timestep_csv_names  # noqa


def main(data_root_dir: str, max_rainfall_threshold: int = 5):
    save_dir_path = os.path.join(
        Path(data_root_dir).parent, "poteka-pipeline-pytorch/preprocess/src"
    )
    with open(os.path.join(save_dir_path, "test_dataset.json"), "r") as f:
        test_cases = json.load(f)
    test_case_dates = get_test_case_dates(test_cases)
    rainfalls_df = search_rainfalls(data_root_dir, max_rainfall_threshold)
    rainfalls_df = rainfalls_df.loc[~rainfalls_df["date"].isin(test_case_dates)]
    rainfalls_df.to_csv(os.path.join(save_dir_path, "train_dataset.csv"))


def get_test_case_dates(test_cases: Dict) -> List[str]:
    dates = []
    for samples in test_cases.values():
        for sample_name in samples.keys():
            dates.append(samples[sample_name]["0"]["date"])
    return dates


class RainfallInfo(BaseModel):
    date: str = None
    max_rainfall: float = 0.0
    start_time: str = None
    peak_time: str = None
    end_time: str = None
    duration: int = 0

    def __repr__(self):
        return f"""
        RainfallInfo(
            date={self.date},
            start_time={self.start_time},
            peak_time={self.peak_time},
            end_time={self.end_time},
            max_rainfall={self.max_rainfall},
            duration={self.duration}
        )"""


def list_dir_by_pattern(root_dir_path: str, regex_pattern: str) -> List:
    prog = re.compile(regex_pattern)
    return [f for f in os.listdir(root_dir_path) if prog.match(f) is not None]


def stringify_interger_month(m: int) -> str:
    if m < 1 or m > 12:
        raise ValueError(f"Invalid int for month: {m}")
    return f"{m}" if m >= 10 else f"0{m}"


def search_rainfalls_in_a_day(
    one_day_data_dir: str,
    date: str,
    minimum_rainfall_duration_minutes: int = 70,
    max_rainfall_threshold: int = 5,
) -> List[RainfallInfo]:
    """Get rainfall information from one_day_data
        start_time: str ('1-0')
        peak_time: str
        end_time: str
        max_rainfall: float
        duration: int (/10 minutes)

    Args:
        one_day_data_dir (str): /path/to/dir (like one_day_data/2020/12/2020-12-15)

    Returns:
        list: list of (start_time, peak_time, end_time, max_rainfall, duration)
            NOTE: end_time is nullable if the rainfall do not end.
    """
    minimum_data_length = 40
    _timestep_csv_names = timestep_csv_names(delta=10)
    rainfall_info = RainfallInfo()
    results = []
    for csv_filename in _timestep_csv_names:
        path = os.path.join(one_day_data_dir, csv_filename)
        if not os.path.exists(path):
            path = path.replace(".csv", ".parquet.gzip")
            df = pd.read_parquet(path)
        else:
            df = pd.read_csv(path)
        rainfall_info.date = date
        # store start_time and peak_time, peak_value
        if df["hour-rain"].max() >= max_rainfall_threshold:
            if df["hour-rain"].max() > rainfall_info.max_rainfall:
                rainfall_info.max_rainfall = df["hour-rain"].max()
                rainfall_info.peak_time = csv_filename.replace(".csv", "")
            if rainfall_info.start_time is None:
                rainfall_info.start_time = csv_filename.replace(".csv", "")
        # rain ends and store information
        if rainfall_info.start_time is not None and df["hour-rain"].max() < 5:
            rainfall_info.end_time = csv_filename.replace(".csv", "")
            duration = _timestep_csv_names.index(
                rainfall_info.end_time + ".csv"
            ) - _timestep_csv_names.index(rainfall_info.start_time + ".csv")
            duration *= 10
            if duration > minimum_rainfall_duration_minutes:
                rainfall_info.duration = duration
                results.append(rainfall_info)
            # re initialize information
            rainfall_info = RainfallInfo()
        # Check if it has enough data
        if len(df.index) < minimum_data_length:
            minimum_data_length = len(df.index)
    # rain do not end in the day
    if (
        minimum_data_length > 0
        and rainfall_info.start_time is not None
        and rainfall_info.end_time is None
    ):
        rainfall_info.end_time = csv_filename.replace(".csv", "")
        duration = _timestep_csv_names.index(
            rainfall_info.end_time + ".csv"
        ) - _timestep_csv_names.index(rainfall_info.start_time + ".csv")
        duration *= 10
        if duration > minimum_rainfall_duration_minutes:
            rainfall_info.duration = duration
            results.append(rainfall_info)
    return results


def search_rainfalls(
    data_root_dir: str, max_rainfall_threshold: int = 5
) -> pd.DataFrame:
    years = ["2019", "2020"]
    monthes = [stringify_interger_month(m) for m in range(1, 13)]
    root_dir = os.path.join(data_root_dir, "one_day_data")
    rainfall_infos: List[RainfallInfo] = []
    for year, month in itertools.product(years, monthes):
        dir_path = os.path.join(root_dir, year, month)
        if not os.path.exists(dir_path):
            continue
        for date in list_dir_by_pattern(dir_path, "\d{4}-\d{2}-\d{2}"):  # noqa: W605
            rainfall_infos += search_rainfalls_in_a_day(
                os.path.join(dir_path, date),
                date,
                max_rainfall_threshold=max_rainfall_threshold,
            )

    df = pd.DataFrame(
        {
            "date": [info.date for info in rainfall_infos],
            "start_time": [info.start_time for info in rainfall_infos],
            "peak_time": [info.peak_time for info in rainfall_infos],
            "end_time": [info.end_time for info in rainfall_infos],
            "duration minutes": [info.duration for info in rainfall_infos],
            "max_rainfall": [info.max_rainfall for info in rainfall_infos],
        }
    )
    df = df.sort_values(by="max_rainfall", ascending=False)
    df.reset_index(inplace=True)
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="process rain data.")
    parser.add_argument(
        "--data_root_dir",
        type=str,
        default="../../../data",
        help="The root path of the data directory",
    )
    parser.add_argument(
        "--max_rainfall_threshold",
        type=int,
        default=5,
        help="The max rainfall amount for training dataset.",
    )
    args = parser.parse_args()
    main(args.data_root_dir, args.max_rainfall_threshold)
