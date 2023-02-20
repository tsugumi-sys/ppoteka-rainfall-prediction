import argparse
import multiprocessing
import os
import re
from datetime import datetime, timedelta
from typing import List

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from tqdm import tqdm


# Create the list of the every 1 minute datetime of a day
# Parameter
# ======================
# start: datetime of start of the day e.g. datetime(2020, 7, 21, 0)
# end: datetime of the end of the day e.g. datetime(2020, 7, 21, 23, 59)
# delta: integer how many times steps to next index e.g. 1
def datetime_range(start, end, delta):
    current = start
    while current <= end:
        yield current
        current += delta


# Create Formated datetime list
# Parameter
# ========================
# year: the number of the year
# month: ...
# date: ...
def create_time_list(year, month, date):
    dts = [
        dt.strftime("%Y-%m-%d T%H:%M Z")
        for dt in datetime_range(
            datetime(year, month, date, 0),
            datetime(year, month, date, 23, 59),
            timedelta(minutes=1),
        )
    ]
    return dts


# Find weather values from 'RAF 0000' -> 0.0
# Parameter
# =========================
# x: the value of cells of csv file
def find_value(x):
    result = re.findall("\d+", str(x))
    if len(result) > 1:
        return result[-1]
    elif len(result) == 1:
        return result[0]
    else:
        return np.nan


# Format 14 digir number to datetime
# Parameter
# ==========================
# x: the value of cells of csv file
def format_datetime(x):
    x = str(x)
    year = int(x[:4])
    month = int(x[4:6])
    date = int(x[6:8])
    hour = int(x[8:10])
    minute = int(x[10:12])
    dt = datetime(year, month, date, hour, minute)
    return dt.strftime("%Y-%m-%d T%H:%M Z")


# Find correct format csv file of the data
# Parameter
# ==========================
# path: {string} the path of the directory that the csv file is in it
# output: {string} the name of the csv file
def find_csv_file(path):
    files = os.listdir(path)
    result = []
    for file in files:
        for item in re.findall("^weather.+.csv$", str(file)):
            result.append(item)
    if len(result) > 1:
        return result[-1]
    else:
        return result[0]
    return result


def extract_data(csv_file_path: str, save_dir_path: str) -> None:
    cols = [
        "Datetime",
        "RAF",
        "RA1",
        "RI1",
        "ERA",
        "CRA",
        "AT1",
        "RH1",
        "PRS",
        "SLP",
        "WD1",
        "WDM",
        "WS1",
        "WSM",
        "WND",
        "WNS",
        "SOL",
        "WET",
        "WBG",
        "WEA",
    ]

    # magnification of weather values crresponding to `cols`
    magni = [10, 10, 10, 10, 10, 10, 10, 10, 10, 1, 1, 10, 10, 1, 10, 1, 1, 10, 1]

    df_before = pd.read_csv(csv_file_path, header=None, error_bad_lines=False,)

    # Extract number of the data from the cell's value
    for col in cols:
        for col_def in df_before.columns:
            if col in str(df_before[col_def].values[0]):
                df_before[col] = df_before[col_def].apply(find_value)
                df_before[col] = df_before[col].apply(float)
    df_before.dropna(inplace=True)

    if df_before.values.size == 0:
        return None

    # Extract datetime and format correctlly
    for col in df_before.columns:
        val = df_before[col].values[0]
        check = find_value(str(val).replace(".0", ""))
        if len(check) == 14:
            y = int(str(val)[:4])
            m = int(str(val)[4:6])
            d = int(str(val)[6:8])
            df_before["Datetime"] = df_before[col].apply(str)
            df_before["Datetime"] = df_before["Datetime"].apply(format_datetime)

    df_before = df_before[cols]
    df_before = df_before.set_index("Datetime")

    # Rescale data
    for i in range(len(cols[1:])):
        col = cols[i + 1]
        mag = magni[i]
        df_before[col] = df_before[col] / mag

    # Reset index if there is a lack of the data
    time_list = create_time_list(y, m, d)
    df_before = df_before.drop_duplicates()
    df = df_before.reindex(time_list)

    os.makedirs(save_dir_path, exist_ok=True)
    df.to_csv(os.path.join(save_dir_path, "data.csv"))


def main(target_years: List[str], data_dir_path: str, n_jobs: int):
    root_path = os.path.join(data_dir_path, "poteka-raw-data")
    if not os.path.exists(root_path):
        raise ValueError(
            f"{root_path} does not exist. "
            "Place raw P-POTEKA data in this path before this process."
        )

    monthes = ["01", "02", "03", "04", "05", "06", "07", "08", "09", "10", "11", "12"]
    args_extract_data = []
    for observation_point in os.listdir(root_path):
        for year in target_years:
            for month in monthes:
                path = os.path.join(root_path, observation_point, f"{year}/{month}")
                if not os.path.exists(path):
                    # warnings.warn(f"{path} does not exist.")
                    continue

                for date in os.listdir(path):
                    arg = {}
                    csv_dir_path = os.path.join(path, f"{date}/Weather")
                    if not os.path.exists(csv_dir_path):
                        # warnings.warn(f"{csv_path} doest not exits.")
                        continue

                    if len(os.listdir(csv_dir_path)) == 0:
                        # warnings.warn(f"The directory {csv_path} is empty.")
                        continue

                    file_name = find_csv_file(csv_dir_path)
                    arg["csv_file_path"] = os.path.join(csv_dir_path, file_name)
                    arg["save_dir_path"] = os.path.join(
                        data_dir_path,
                        "cleaned-data",
                        observation_point,
                        year,
                        month,
                        date,
                    )
                    args_extract_data.append(arg)

        Parallel(n_jobs=n_jobs)(
            delayed(extract_data)(
                csv_file_path=arg["csv_file_path"], save_dir_path=arg["save_dir_path"],
            )
            for arg in tqdm(args_extract_data)
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract and rescale the raw P-POTEKA data."
    )
    parser.add_argument(
        "--data_dir_path",
        type=str,
        default="../../../data",
        help="The root path of the data directory.",
    )
    parser.add_argument(
        "--n_jobs", type=int, default=10, help="CPU counts for mutiprocessing.",
    )
    parser.add_argument(
        "--target_years",
        type=str,
        default="2019/2020",
        help="Target years to process. Use slash for multipce years.",
    )
    args = parser.parse_args()
    target_years = args.target_years.split("/")
    for y in target_years:
        print(y)
        if re.match(r"^\d{4}$", y) is None:
            raise ValueError(
                "Invalid `--target_years` format. Use slash like `2019/2020`, "
                f"instead of {args.target_years}."
            )
    n_jobs = args.n_jobs
    max_cpus = multiprocessing.cpu_count()
    if n_jobs > multiprocessing.cpu_count():
        n_jobs = max_cpus - 1
    main(
        target_years=target_years, data_dir_path=args.data_dir_path, n_jobs=n_jobs,
    )
