import argparse
import multiprocessing
import os

import pandas as pd
from joblib import Parallel, delayed
from sklearn.impute import SimpleImputer
from tqdm import tqdm


def impute_data(csv_file_path: str, save_dir_path: str) -> None:
    imp = SimpleImputer(strategy="mean")
    rain_cols = ["RAF", "RA1", "RI1", "ERA", "CRA"]
    df = pd.read_csv(csv_file_path, index_col="Datetime")
    if len(df) != 1440:
        # The lacking data is too much
        return None

    imp_df = df.copy()
    imp_df = imp_df.fillna(method="bfill", axis=0, limit=3)
    imp_df = imp_df.fillna(method="ffill", axis=0, limit=3)
    imp_df[rain_cols] = imp_df[rain_cols].fillna(0)
    imp_df = pd.DataFrame(imp.fit_transform(imp_df))
    imp_df.columns = df.columns
    imp_df.index = df.index

    # check nan value
    # total_missing = df.isnull().sum().sum()
    # total_missing_imputed = imp_df.isnull().sum().sum()
    # total_cells = np.product(imp_df.shape)
    # print("Percentage of nan cells")
    # print(
    #     "Before: ",
    #     (total_missing / total_cells) * 100,
    # )
    # print(
    #     "After: ",
    #     (total_missing_imputed / total_cells) * 100,
    # )

    # save
    os.makedirs(save_dir_path, exist_ok=True)
    imp_df.to_csv(os.path.join(save_dir_path, "data.csv"))


def main(data_dir_path: str, n_jobs: int):
    cleaned_data_dir = os.path.join(data_dir_path, "cleaned-data")
    args_impute_data = []
    ob_folders = os.listdir(cleaned_data_dir)
    for ob_folder in ob_folders:
        folders = os.listdir(os.path.join(cleaned_data_dir, ob_folder))
        if len(folders) == 0:
            continue
        for year_folder in folders:
            month_folders = os.listdir(
                os.path.join(cleaned_data_dir, ob_folder, year_folder)
            )
            if len(month_folders) == 0:
                continue
            for month_folder in month_folders:
                data_folders = os.listdir(
                    os.path.join(cleaned_data_dir, ob_folder, year_folder, month_folder)
                )
                if len(data_folders) == 0:
                    continue
                for data_folder in data_folders:
                    arg = {}
                    data_file = os.path.join(
                        cleaned_data_dir,
                        ob_folder,
                        year_folder,
                        month_folder,
                        data_folder,
                        "data.csv",
                    )

                    if not os.path.exists(data_file):
                        continue

                    save_dir_path = os.path.join(
                        data_dir_path,
                        "imputed-data",
                        ob_folder,
                        year_folder,
                        month_folder,
                        data_folder,
                    )
                    arg["csv_file_path"] = data_file
                    arg["save_dir_path"] = save_dir_path
                    args_impute_data.append(arg)

        Parallel(n_jobs=n_jobs)(
            delayed(impute_data)(
                csv_file_path=arg["csv_file_path"], save_dir_path=arg["save_dir_path"],
            )
            for arg in tqdm(args_impute_data)
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
