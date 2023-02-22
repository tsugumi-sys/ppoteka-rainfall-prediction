import argparse
import multiprocessing
import os
from datetime import date, timedelta
from typing import Optional

import pandas as pd
from joblib import Parallel, delayed
from tqdm import tqdm


def calc_hourly_rain(
    csv_file_path: str,
    save_dir_path: str,
    yesterday_csv_file_path: Optional[str] = None,
) -> None:
    df_today = pd.read_csv(csv_file_path, index_col="Datetime")
    df = df_today.copy()
    if yesterday_csv_file_path is not None:
        df_yesterday = pd.read_csv(yesterday_csv_file_path, index_col="Datetime")
        df = pd.concat([df_yesterday, df])
    df["hour-rain"] = df["RAF"].rolling(60, min_periods=1).sum()
    df_today["hour-rain"] = df["hour-rain"].loc[df_today.index]

    os.makedirs(save_dir_path, exist_ok=True)
    df_today.to_csv(os.path.join(save_dir_path, "data.csv"))


def main(data_dir_path: str, n_jobs: int):
    imputed_data_dir = os.path.join(data_dir_path, "imputed-data")
    args_calc_hourly_rain = []
    ob_folders = os.listdir(imputed_data_dir)
    for ob_folder in ob_folders:
        folders = os.listdir(os.path.join(imputed_data_dir, ob_folder))
        if len(folders) == 0:
            continue
        for year_folder in folders:
            month_folders = os.listdir(
                os.path.join(imputed_data_dir, ob_folder, year_folder)
            )
            if len(month_folders) == 0:
                continue
            for month_folder in month_folders:
                data_folders = os.listdir(
                    os.path.join(imputed_data_dir, ob_folder, year_folder, month_folder)
                )
                if len(data_folders) == 0:
                    continue
                for data_folder in data_folders:
                    arg = {}
                    csv_file_path = os.path.join(
                        imputed_data_dir,
                        ob_folder,
                        year_folder,
                        month_folder,
                        data_folder,
                        "data.csv",
                    )
                    if not os.path.exists(csv_file_path):
                        continue

                    yesterday_csv_file_path = os.path.join(
                        imputed_data_dir,
                        ob_folder,
                        year_folder,
                        month_folder,
                        str(date.fromisoformat(data_folder) - timedelta(days=1)),
                        "data.csv",
                    )
                    if not os.path.exists(yesterday_csv_file_path):
                        yesterday_csv_file_path = None

                    save_dir_path = os.path.join(
                        data_dir_path,
                        "accumulated-raf-data",
                        ob_folder,
                        year_folder,
                        month_folder,
                        data_folder,
                    )
                    arg["csv_file_path"] = csv_file_path
                    arg["save_dir_path"] = save_dir_path
                    arg["yesterday_csv_file_path"] = yesterday_csv_file_path
                    args_calc_hourly_rain.append(arg)
    Parallel(n_jobs=n_jobs)(
        delayed(calc_hourly_rain)(
            csv_file_path=arg["csv_file_path"],
            save_dir_path=arg["save_dir_path"],
            yesterday_csv_file_path=arg["yesterday_csv_file_path"],
        )
        for arg in tqdm(args_calc_hourly_rain)
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="process humidity data.")
    parser.add_argument(
        "--data_dir_path",
        type=str,
        default="../../../data",
        help="The root path of the data directory.",
    )
    parser.add_argument(
        "--n_jobs", type=int, default=10, help="CPU counts for mutiprocessing.",
    )
    args = parser.parse_args()
    n_jobs = args.n_jobs
    max_cpus = multiprocessing.cpu_count()
    if n_jobs > multiprocessing.cpu_count():
        n_jobs = max_cpus - 1
    main(args.data_dir_path, n_jobs)
