import unittest

import pandas as pd

from common.utils import timestep_csv_names
from common.config import WEATHER_PARAMS
from preprocess.src.extract_data import get_test_data_files, get_train_data_files


class TestPreprocessExtractdata(unittest.TestCase):
    def test_get_train_data_files(self):
        train_list_df = pd.DataFrame(
            {
                "date": ["2020-01-01"],
                "start_time": ["10-0"],
                "end_time": ["10-0"],
            }
        )
        input_seq_length, label_seq_length = 6, 6
        expected_params = WEATHER_PARAMS.valid_params()
        expected_params = [i for i in expected_params if i != "wind"]
        train_data_files = get_train_data_files(
            project_root_dir_path="/tmp/",
            train_list_df=train_list_df,
            input_parameters=expected_params,
            time_step_minutes=10,
            time_slides_delta=3,
            input_seq_length=input_seq_length,
            label_seq_length=label_seq_length,
        )
        self.assertIsInstance(train_data_files, list)
        _timestep_csv_names = timestep_csv_names(time_step_minutes=10)
        # input wind param, u-wind and v-wind loads
        for sample in train_data_files:
            with self.subTest(sample=sample):
                self.assertEqual(sorted(list(sample.keys())), sorted(expected_params))
                # Each params has input and label key

                for p in expected_params:
                    with self.subTest(param=p):
                        param_data_files = sample[p]
                        self.assertEqual(sorted(list(param_data_files.keys())), sorted(["input", "label"]))
                        # input length is 6, label length is 6
                        self.assertEqual(len(param_data_files["input"]), 6)
                        self.assertEqual(len(param_data_files["label"]), 6)
                        # test the filenames are correctly ranged
                        # [NOTE]: U, V wind filename is U.csv and V.csv.
                        input_start_idx = _timestep_csv_names.index(
                            param_data_files["input"][0]
                            .split("/")[-1]
                            .replace("U.csv", ".csv")
                            .replace("V.csv", ".csv")
                        )
                        label_start_idx = _timestep_csv_names.index(
                            param_data_files["label"][0]
                            .split("/")[-1]
                            .replace("U.csv", ".csv")
                            .replace("V.csv", ".csv")
                        )
                        self.assertEqual(
                            [
                                filepath.split("/")[-1].replace("U.csv", ".csv").replace("V.csv", ".csv")
                                for filepath in param_data_files["input"]
                            ],
                            _timestep_csv_names[input_start_idx : input_start_idx + input_seq_length],  # noqa: E203
                        )
                        self.assertEqual(
                            [
                                filepath.split("/")[-1].replace("U.csv", ".csv").replace("V.csv", ".csv")
                                for filepath in param_data_files["label"]
                            ],
                            _timestep_csv_names[label_start_idx : label_start_idx + label_seq_length],  # noqa: E203
                        )

    def test_get_test_data_files(self):
        test_data_list = {
            "TC_case": {
                "sample1": {
                    0: {"date": "2020-10-12", "start": "5-0.csv"},
                }
            },
            "NOT_TC_case": {
                "sample1": {
                    0: {"date": "2020-07-04", "start": "6-0.csv"},
                }
            },
        }
        expected_params = WEATHER_PARAMS.valid_params()
        expected_params = [i for i in expected_params if i != "wind"]
        input_seq_length, label_seq_length = 6, 6
        test_data_files = get_test_data_files(
            project_root_dir_path="/tmp/",
            test_data_list=test_data_list,
            input_parameters=expected_params,
            time_step_minutes=10,
            input_seq_length=input_seq_length,
            label_seq_length=label_seq_length,
        )
        # Keys should be sample names like TC_case_{date}_{start} without .csv
        expected_key_names = []
        for key in test_data_list.keys():
            for sample_name in test_data_list[key].keys():
                for case_idx in test_data_list[key][sample_name]:
                    case = test_data_list[key][sample_name][case_idx]
                    start = case["start"].replace(".csv", "")
                    date = case["date"]
                    expected_key_names += [f"{key}_{date}_{start}_start"]
        self.assertEqual(sorted(list(test_data_files.keys())), sorted(expected_key_names))
        # input wind param, u-wind and v-wind loads
        expected_params = WEATHER_PARAMS.valid_params()
        expected_params = [i for i in expected_params if i != "wind"]
        # test_data_files should has date and start
        test_data_expected_params = expected_params.copy()
        test_data_expected_params += ["date", "start"]
        _timestep_csv_names = timestep_csv_names(time_step_minutes=10)
        for case in test_data_files.values():
            # each caase has each parameters
            with self.subTest(case=case):
                self.assertEqual(sorted(list(case.keys())), sorted(test_data_expected_params))

                for param in expected_params:
                    with self.subTest(param=param):
                        param_data_files = case[param]
                        self.assertEqual(sorted(list(param_data_files.keys())), sorted(["input", "label"]))
                        self.assertEqual(len(param_data_files["input"]), 6)
                        self.assertEqual(len(param_data_files["label"]), 6)
                        # test the filenames are correctly ranged
                        # [NOTE]: U, V wind filename is U.csv and V.csv.
                        pred_start_idx = _timestep_csv_names.index(case["start"])
                        self.assertEqual(
                            [
                                filepath.split("/")[-1].replace("U.csv", ".csv").replace("V.csv", ".csv")
                                for filepath in param_data_files["input"]
                            ],
                            _timestep_csv_names[pred_start_idx - input_seq_length : pred_start_idx],  # noqa: E203
                        )
                        self.assertEqual(
                            [
                                filepath.split("/")[-1].replace("U.csv", ".csv").replace("V.csv", ".csv")
                                for filepath in param_data_files["label"]
                            ],
                            _timestep_csv_names[pred_start_idx : pred_start_idx + label_seq_length],  # noqa: E203
                        )
