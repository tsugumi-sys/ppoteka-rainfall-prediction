import argparse
import os
from datetime import date, timedelta

import pandas as pd


def main(data_dir_path: str):
    folder_path = os.path.join(data_dir_path, "imputed-data")
    ob_folders = os.listdir(folder_path)
    for ob_folder in ob_folders:
        folders = os.listdir(folder_path + f"/{ob_folder}")
        if folders:
            for year_folder in folders:
                month_folders = os.listdir(folder_path + f"/{ob_folder}/{year_folder}")
                if month_folders:
                    for month_folder in month_folders:
                        data_folders = os.listdir(
                            folder_path + f"/{ob_folder}/{year_folder}/{month_folder}"
                        )
                        if data_folders:
                            for data_folder in data_folders:
                                data_file = (
                                    folder_path
                                    + f"/{ob_folder}/{year_folder}/{month_folder}/{data_folder}/data.csv"
                                )
                                if os.path.exists(data_file):
                                    print("-" * 60)
                                    print(data_file)
                                    today = data_folder
                                    yesterday = date.fromisoformat(today) - timedelta(
                                        days=1
                                    )
                                    df_today = pd.read_csv(
                                        data_file, index_col="Datetime"
                                    )
                                    yesterday_path = (
                                        folder_path
                                        + f"/{ob_folder}/{year_folder}/{month_folder}/{yesterday}/data.csv"
                                    )
                                    if os.path.exists(yesterday_path):
                                        df_yesterday = pd.read_csv(
                                            yesterday_path, index_col="Datetime"
                                        )
                                        df = pd.concat([df_yesterday, df_today])
                                        df["hour-rain"] = (
                                            df["RAF"].rolling(60, min_periods=1).sum()
                                        )
                                        df_today["hour-rain"] = df["hour-rain"].loc[
                                            df_today.index
                                        ]

                                        # save
                                        save_path = os.path.join(
                                            data_dir_path, "accumulated-raf-data"
                                        )
                                        f = [
                                            ob_folder,
                                            year_folder,
                                            month_folder,
                                            data_folder,
                                        ]
                                        for i in f:
                                            save_path = save_path + f"/{i}"
                                            print(save_path)
                                            if not os.path.exists(save_path):
                                                os.mkdir(save_path)
                                        save_path = save_path + "/data.csv"
                                        df_today.to_csv(save_path)
                                    else:
                                        df_today["hour-rain"] = (
                                            df_today["RAF"]
                                            .rolling(60, min_periods=1)
                                            .sum()
                                        )
                                        # save
                                        save_path = "../../../data/accumulated-raf-data"
                                        f = [
                                            ob_folder,
                                            year_folder,
                                            month_folder,
                                            data_folder,
                                        ]
                                        for i in f:
                                            save_path = save_path + f"/{i}"
                                            print(save_path)
                                            if not os.path.exists(save_path):
                                                os.mkdir(save_path)
                                        save_path = save_path + "/data.csv"
                                        df_today.to_csv(save_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="process humidity data.")
    parser.add_argument(
        "--data_dir_path",
        type=str,
        default="../../../data",
        help="The root path of the data directory.",
    )
    args = parser.parse_args()
    main(args.data_root_dir)