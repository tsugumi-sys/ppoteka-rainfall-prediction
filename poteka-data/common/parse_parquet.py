import pandas as pd
from utils import timestep_csv_names
import os

def parse_parquet(df, save_path):
    df.to_parquet(
        save_path,
        engine="pyarrow",
        # compression="gzip",
    )


if __name__ == "__main__":
    root_dir = "../../data/one_day_data"
    _timestep_csv_names = timestep_csv_names(delta=2)
    
    for year in os.listdir(root_dir):
        for month in os.listdir(root_dir + f"/{year}"):
            for date in os.listdir(root_dir + f"/{year}/{month}"):
                max_rainfall = 0
                
                assert len(os.listdir(os.path.join(root_dir, year, month, date))) == len(_timestep_csv_names)
                for csv_filename in _timestep_csv_names:
                    csv_path = os.path.join(
                        root_dir,
                        year, month, date, csv_filename
                    )
                    parquet_path = csv_path.replace(".csv", ".parquet.gzip")

                    assert (not os.path.exists(csv_path)) and (os.path.exists(parquet_path))

                    # df = pd.read_csv(csv_path)
                    # parse_parquet(df, parquet_path)
                    # os.remove(csv_path)