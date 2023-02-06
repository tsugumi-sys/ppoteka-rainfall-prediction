import argparse
import json
import os
from pathlib import Path


def main(data_root_dir: str, timestep_delta: int):
    # Modify the dictinary below for customizing test cases.
    test_cases = {
        "TC_case": {
            "sample1": {
                "0": {"date": "2020-10-12", "start": "5-0.csv"},
                "1": {"date": "2020-10-12", "start": "5-20.csv"},
                "2": {"date": "2020-10-12", "start": "5-40.csv"},
                "3": {"date": "2020-10-12", "start": "6-0.csv"},
                "4": {"date": "2020-10-12", "start": "6-20.csv"},
                "5": {"date": "2020-10-12", "start": "6-40.csv"},
                "6": {"date": "2020-10-12", "start": "7-0.csv"},
                "7": {"date": "2020-10-12", "start": "7-20.csv"},
                "8": {"date": "2020-10-12", "start": "7-40.csv"},
                "9": {"date": "2020-10-12", "start": "8-0.csv"},
                "10": {"date": "2020-10-12", "start": "8-20.csv"},
                "11": {"date": "2020-10-12", "start": "8-40.csv"},
                "12": {"date": "2020-10-12", "start": "9-0.csv"},
                "13": {"date": "2020-10-12", "start": "9-20.csv"},
                "14": {"date": "2020-10-12", "start": "9-40.csv"},
            },
            "sample2": {
                "0": {"date": "2020-09-14", "start": "4-0.csv"},
                "1": {"date": "2020-09-14", "start": "4-20.csv"},
                "2": {"date": "2020-09-14", "start": "4-40.csv"},
                "3": {"date": "2020-09-14", "start": "5-0.csv"},
                "4": {"date": "2020-09-14", "start": "5-20.csv"},
                "5": {"date": "2020-09-14", "start": "5-40.csv"},
                "6": {"date": "2020-09-14", "start": "6-0.csv"},
                "7": {"date": "2020-09-14", "start": "6-20.csv"},
                "8": {"date": "2020-09-14", "start": "6-40.csv"},
            },
            "sample3": {
                "0": {"date": "2020-08-07", "start": "4-0.csv"},
                "1": {"date": "2020-08-07", "start": "4-20.csv"},
                "2": {"date": "2020-08-07", "start": "4-40.csv"},
                "3": {"date": "2020-08-07", "start": "5-0.csv"},
                "4": {"date": "2020-08-07", "start": "5-20.csv"},
                "5": {"date": "2020-08-07", "start": "5-40.csv"},
                "6": {"date": "2020-08-07", "start": "6-0.csv"},
                "7": {"date": "2020-08-07", "start": "6-20.csv"},
                "8": {"date": "2020-08-07", "start": "6-40.csv"},
                "9": {"date": "2020-08-07", "start": "7-0.csv"},
                "10": {"date": "2020-08-07", "start": "7-20.csv"},
                "11": {"date": "2020-08-07", "start": "7-40.csv"},
            },
        },
        "NOT_TC_case": {
            "sample1": {
                "0": {"date": "2020-07-04", "start": "6-0.csv"},
                "1": {"date": "2020-07-04", "start": "6-20.csv"},
                "2": {"date": "2020-07-04", "start": "6-40.csv"},
                "3": {"date": "2020-07-04", "start": "7-0.csv"},
                "4": {"date": "2020-07-04", "start": "7-20.csv"},
                "5": {"date": "2020-07-04", "start": "7-40.csv"},
                "6": {"date": "2020-07-04", "start": "8-0.csv"},
                "7": {"date": "2020-07-04", "start": "8-20.csv"},
                "8": {"date": "2020-07-04", "start": "8-40.csv"},
                "9": {"date": "2020-07-04", "start": "9-0.csv"},
                "10": {"date": "2020-07-04", "start": "9-20.csv"},
                "11": {"date": "2020-07-04", "start": "9-40.csv"},
                "12": {"date": "2020-07-04", "start": "10-0.csv"},
                "13": {"date": "2020-07-04", "start": "10-20.csv"},
                "14": {"date": "2020-07-04", "start": "10-40.csv"},
            },
            "sample2": {
                "0": {"date": "2019-10-04", "start": "4-0.csv"},
                "1": {"date": "2019-10-04", "start": "4-20.csv"},
                "2": {"date": "2019-10-04", "start": "4-40.csv"},
                "3": {"date": "2019-10-04", "start": "5-0.csv"},
                "4": {"date": "2019-10-04", "start": "5-20.csv"},
                "5": {"date": "2019-10-04", "start": "5-40.csv"},
                "6": {"date": "2019-10-04", "start": "6-0.csv"},
                "7": {"date": "2019-10-04", "start": "6-20.csv"},
                "8": {"date": "2019-10-04", "start": "6-40.csv"},
                "9": {"date": "2019-10-04", "start": "7-0.csv"},
                "10": {"date": "2019-10-04", "start": "7-20.csv"},
                "11": {"date": "2019-10-04", "start": "7-40.csv"},
            },
            "sample3": {
                "0": {"date": "2019-10-12", "start": "8-0.csv"},
                "1": {"date": "2019-10-12", "start": "8-20.csv"},
                "2": {"date": "2019-10-12", "start": "8-40.csv"},
                "3": {"date": "2019-10-12", "start": "9-0.csv"},
                "4": {"date": "2019-10-12", "start": "9-20.csv"},
                "5": {"date": "2019-10-12", "start": "9-40.csv"},
                "6": {"date": "2019-10-12", "start": "10-0.csv"},
                "7": {"date": "2019-10-12", "start": "10-20.csv"},
                "8": {"date": "2019-10-12", "start": "10-40.csv"},
            },
        },
    }
    save_dir_path = os.path.join(Path(data_root_dir).parent, "poteka-pipeline-pytorch/preprocess/src")
    with open(os.path.join(save_dir_path, "test_dataset.json"), "w") as f:
        json.dump(test_cases, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="process rain data.")
    parser.add_argument(
        "--data_root_dir",
        type=str,
        default="../../../data",
        help="The root path of the data directory",
    )
    parser.add_argument(
        "--timestep_delta",
        type=int,
        default=10,
        help="time step delta (minutes).",
    )
    args = parser.parse_args()
    main(args.data_root_dir, args.timestep_delta)
