import sys
import json

import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor, kernels

sys.path.append(".")
from evaluate.src.interpolator.interpolator_interface import InterpolatorInterface


class RainInterpolator(InterpolatorInterface):
    def _init__(self) -> None:
        pass

    def interpolate(self, ndarray: np.ndarray, observation_point_file_path: str) -> np.ndarray:
        if ndarray.max() > 1 or ndarray.min() < 0:
            raise ValueError(f"The scale of the given ndarray is invalid (max: {ndarray.max()}, min: {ndarray.min()}). The scale should be [0, 1]")

        if ndarray.ndim != 1:
            raise ValueError(f"The given ndarray is invalid dimention (dim: {ndarray.ndim}). The dimention should be 1.")

        with open(observation_point_file_path, "r") as f:
            ob_point_data = json.load(f)

        ob_point_lons = [val["longitude"] for val in ob_point_data.values()]
        ob_point_lats = [val["latitude"] for val in ob_point_data.values()]

        grid_coordinate = np.mgrid[120.90:121.150:50j, 14.350:14.760:50j]

        kernel = kernels.ConstantKernel(1, (1e-5, 1e5)) * kernels.RBF(1, (1e-5, 1e5))
        gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=15, random_state=123)

        x = np.column_stack([ob_point_lons, ob_point_lats])
        gp.fit(x, ndarray)

        y_pred, _ = gp.predict(grid_coordinate.reshape(2, -1).T, return_std=True)
        grid_data = np.reshape(y_pred, (50, 50)).T
        grid_data = np.flipud(grid_data)

        grid_data = np.where(grid_data > 0, grid_data, 0)
        grid_data = np.where(grid_data > 1, 1, grid_data)

        return grid_data.astype(np.float32)

    def interpolate_with_gpu(self, ndarray: np.ndarray, observation_point_file_path: str) -> np.ndarray:
        pass
