import os
from pathlib import Path


class DATAFOLDER:
    # Change your path.

    # current_dir = os.path.abspath("../")
    path = Path(__file__).resolve()
    root_dir = path.parent.absolute()
    # Get poteka_data_analysis root directory path
    root_dir = root_dir.parent.absolute()
    # Get data folder path
    root_dir = root_dir.parent.absolute()
    data_root_path = os.path.join(root_dir, "data/")
