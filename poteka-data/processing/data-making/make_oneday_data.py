import argparse
import itertools
import json
import multiprocessing
import os
import sys
from datetime import datetime
from logging import INFO, StreamHandler, basicConfig, captureWarnings, getLogger
from pathlib import Path
from typing import Dict

import pandas as pd
from joblib import Parallel, delayed

sys.path.append(".")
from common.send_info import send_line  # noqa
from common.utils import create_time_list, make_dates  # noqa

logger = getLogger(__name__)
logger.setLevel(INFO)
captureWarnings(True)
logger.addHandler(StreamHandler(sys.stdout))


def make_oneday_data(
    data_root_dir: str,
    save_dir_path: str,
    year: str,
    month: str,
    date: str,
    delta: int,
    obpoint_locations: Dict,
) -> None:
    print("Creating", year, month, date, "data ...")
    folder_path = os.path.join(data_root_dir, "accumulated-raf-data")

    cols = ["LON", "LAT", "hour-rain", "AT1", "RH1", "SOL", "WD1", "WS1", "PRS", "SLP"]
    data_cols = ["hour-rain", "AT1", "RH1", "SOL", "WD1", "WS1", "PRS", "SLP"]
    count = 0
    for ob_point in os.listdir(folder_path):
        path = os.path.join(
            folder_path, ob_point, year, month, f"{year}-{month}-{date}", "data.csv"
        )
        print(path)
        if os.path.exists(path):
            count += 1

    if count == 0:
        # The folder does not exist.
        logger.error(f"{year}/{month}/{date} does not exist.")
        return

    # Create One day Data
    time_list = create_time_list(int(year), int(month), int(date), delta)
    print(f"{year}-{month}-{date} ", count)
    ob_data = []
    ob_names = []
    ob_lon_lat = []
    for ob_point in os.listdir(folder_path):
        path = os.path.join(
            folder_path, ob_point, year, month, f"{year}-{month}-{date}", "data.csv"
        )
        if os.path.exists(path):
            ob_names.append(ob_point)
            ob_df = pd.read_csv(path, index_col="Datetime")
            ob_data.append(ob_df)
            ob_lon_lat.append(
                [
                    obpoint_locations[ob_point]["longitude"],
                    obpoint_locations[ob_point]["latitude"],
                ]
            )

    # Save Folder
    save_dir_path = os.path.join(save_dir_path, year, month, f"{year}-{month}-{date}")
    os.makedirs(save_dir_path, exist_ok=True)

    for time in time_list:
        df = pd.DataFrame(index=ob_names, columns=cols)
        datetime_str = datetime.strftime(time, "%Y-%m-%d T%H:%M Z")
        for i in range(len(ob_data)):
            data = ob_data[i]
            ob_name = ob_names[i]
            ob_point = ob_lon_lat[i]
            df.loc[ob_name, "LON"], df.loc[ob_name, "LAT"] = ob_point
            for col in data_cols:
                df.loc[ob_name, col] = data.loc[datetime_str, col]

        df.to_csv(os.path.join(save_dir_path, f"{time.hour}-{time.minute}.csv"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="process oneday data.")
    parser.add_argument(
        "--data_root_dir",
        type=str,
        default="../../../data",
        help="The root path of the data directory",
    )
    parser.add_argument(
        "--log_dir",
        type=str,
        default="log",
        help="The path of a directory to store log files.",
    )
    parser.add_argument(
        "--n_jobs", type=int, default=1, help="The number of cpu cores to use",
    )
    parser.add_argument(
        "--delta", type=int, default=10, help="time step",
    )
    args = parser.parse_args()

    log_dir = args.log_dir
    os.makedirs(log_dir, exist_ok=True)
    basicConfig(
        level=INFO,
        filename=os.path.join(log_dir, "create_oneday_data.log"),
        filemode="w",
        format="%(asctime)s %(levelname)s %(name)s :%(message)s",
    )
    data_root_dir = args.data_root_dir
    save_dir_path = os.path.join(data_root_dir, "one_day_data")
    os.makedirs(save_dir_path, exist_ok=True)

    n_jobs = args.n_jobs
    max_cores = multiprocessing.cpu_count()
    if n_jobs > max_cores:
        n_jobs = max_cores

    delta = args.delta
    years = ["2019", "2020"]
    monthes = ["01", "02", "03", "04", "05", "06", "07", "08", "09", "10", "11", "12"]
    dates = [make_dates(i) for i in range(1, 32)]
    ymds = itertools.product(years, monthes, dates)
    # Load observation point location data
    with open(
        os.path.join(
            Path(data_root_dir).parent,
            "poteka-pipeline-pytorch/common/meta-data/observation_point.json",
        )
    ) as f:
        obpoint_locations = json.load(f)

    logger.info("Process start")
    Parallel(n_jobs=n_jobs)(
        delayed(make_oneday_data)(
            data_root_dir=data_root_dir,
            save_dir_path=save_dir_path,
            year=ymd[0],
            month=ymd[1],
            date=ymd[2],
            delta=delta,
            obpoint_locations=obpoint_locations,
        )
        for ymd in list(ymds)
    )
    logger.info("process finished")
    send_line("Creating oneday data finished.")
