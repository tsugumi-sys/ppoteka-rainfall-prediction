import argparse
import itertools
import multiprocessing
import os
import sys
from datetime import datetime
from logging import INFO, StreamHandler, basicConfig, captureWarnings, getLogger

import pandas as pd
from joblib import Parallel, delayed

sys.path.append(".")
from common.send_info import send_line
from common.utils import create_time_list, make_dates

logger = getLogger(__name__)
logger.setLevel(INFO)
basicConfig(
    level=INFO,
    filename="./dataset/data-making/log/create_oneday_data.log",
    filemode="w",
    format="%(asctime)s %(levelname)s %(name)s :%(message)s",
)
captureWarnings(True)
logger.addHandler(StreamHandler(sys.stdout))


def make_oneday_data(
    data_root_dir: str, year: str, month: str, date: str, delta: int
) -> None:
    print("Creating", year, month, date, "data ...")
    folder_path = os.path.join(data_root_dir, "accumulated-raf-data")
    ob_locations = pd.read_csv(
        "../p-poteka-config/observation_point.csv", index_col="Name"
    )
    cols = ["LON", "LAT", "hour-rain", "AT1", "RH1", "SOL", "WD1", "WS1", "PRS", "SLP"]
    data_cols = ["hour-rain", "AT1", "RH1", "SOL", "WD1", "WS1", "PRS", "SLP"]
    count = 0
    for ob_point in os.listdir(folder_path):
        path = (
            folder_path + f"/{ob_point}/{year}/{month}/{year}-{month}-{date}/data.csv"
        )
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
        path = (
            folder_path + f"/{ob_point}/{year}/{month}/{year}-{month}-{date}/data.csv"
        )
        if os.path.exists(path):
            ob_names.append(ob_point)
            ob_df = pd.read_csv(path, index_col="Datetime")
            ob_data.append(ob_df)
            ob_lon, ob_lat = (
                ob_locations.loc[ob_point, "LON"],
                ob_locations.loc[ob_point, "LAT"],
            )
            ob_lon_lat.append([ob_lon, ob_lat])

    # Save Folder
    save_folder_path = "../data/one_day_data"
    create_folder = [year, month, f"{year}-{month}-{date}"]
    for folder_name in create_folder:
        save_folder_path = save_folder_path + f"/{folder_name}"
        if not os.path.exists(save_folder_path):
            os.mkdir(save_folder_path)

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
        # print(df)

        save_path = save_folder_path + f"/{time.hour}-{time.minute}.csv"
        df.to_csv(save_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="process oneday data.")
    parser.add_argument(
        "--data_root_dir",
        type=str,
        default="../../../data",
        help="The root path of the data directory",
    )
    parser.add_argument(
        "--n_jobs",
        type=int,
        default=1,
        help="The number of cpu cores to use",
    )
    parser.add_argument(
        "--delta",
        type=int,
        default=10,
        help="time step",
    )
    args = parser.parse_args()

    n_jobs = args.n_jobs
    max_cores = multiprocessing.cpu_count()
    if n_jobs > max_cores:
        n_jobs = max_cores

    delta = args.delta
    years = ["2019", "2020"]
    monthes = ["01", "02", "03", "04", "05", "06", "07", "08", "09", "10", "11", "12"]
    dates = [make_dates(i) for i in range(1, 32)]
    ymds = itertools.product(years, monthes, dates)

    logger.info("Process start")
    Parallel(n_jobs=n_jobs)(
        delayed(make_oneday_data)(
            data_root_dir=args.data_root_dir,
            year=ymd[0],
            month=ymd[1],
            date=ymd[2],
            delta=delta,
        )
        for ymd in list(ymds)
    )
    logger.info("process finished")
    send_line("Creating oneday data finished.")
