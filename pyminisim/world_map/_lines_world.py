from typing import Union, Tuple

import numpy as np
from scipy.spatial.distance import cdist

from pyminisim.core import AbstractWorldMapState, AbstractStaticWorldMap


class _LinesWorldState(AbstractWorldMapState):

    def __init__(self, lines: np.ndarray):
        assert len(lines.shape) == 3 and lines.shape[1] == 2 and lines.shape[2] == 2
        super(_LinesWorldState, self).__init__()
        self._lines = lines

    @property
    def lines(self) -> np.ndarray:
        return self._lines


class LinesWorld(AbstractStaticWorldMap):

    def __init__(self, lines: np.ndarray, line_width: float = 0.05):
        """
        :param lines: List of circles in format [(x1, y1), (x2, y2)] (in metres)
        """
        super(LinesWorld, self).__init__(_LinesWorldState(lines.copy()))
        self._line_width = line_width

    def closest_point(self, point: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        assert point.shape == (2,) or (len(point.shape) == 2 and point.shape[1] == 2)
        if point.shape == (2,):
            is_single_point = True
            point = point[np.newaxis, :]
        else:
            is_single_point = False

        lines = self._state.lines
        n_lines = lines.shape[0]
        n_points = point.shape[0]

        ab = lines[:, 1, :] - lines[:, 0, :]  # (n_lines, 2)
        ap = point - lines[:, np.newaxis, 0, :]  # (n_lines, n_points, 2)
        proj = np.einsum("ijk,ik->ij", ap, ab)  # (n_lines, n_points)
        d = proj / (np.linalg.norm(ab, axis=1) ** 2)[:, np.newaxis]  # (n_lines, n_points)
        d = np.clip(d, 0., 1.)
        closest_points = lines[:, np.newaxis, 0, :] + \
                         ab[:, np.newaxis, :] * d[:, :, np.newaxis]  # (n_lines, n_points, 2)

        distances = np.linalg.norm(point - closest_points, axis=-1)
        min_lines_indices = np.argmin(distances, axis=0)
        # closest_points = closest_points[np.argmin(distances, axis=0), :, :]
        result_points = np.empty((n_points, 2))
        result_distances = np.empty(n_points)
        for i in range(n_points):
            result_points[i, :] = closest_points[min_lines_indices[i], i, :]
            result_distances[i] = distances[min_lines_indices[i], i]

        if is_single_point:
            result_points = result_points[0]
            result_distances = result_distances[0]

        return result_points, result_distances

    def closest_distance_to_obstacle(self, point: np.ndarray) -> \
            Union[float, np.ndarray]:
        _, distances = self.closest_point(point)
        return distances

    def is_occupied(self, point: np.ndarray) -> Union[bool, np.ndarray]:
        # TODO: Change occupancy notion in the base class
        assert point.shape == (2,) or (len(point.shape) == 2 and point.shape[1] == 2)
        min_dists = self.closest_distance_to_obstacle(point)
        result = min_dists > self._line_width / 2
        return result

    @property
    def lines(self) -> np.ndarray:
        return self._state.lines.copy()

    @property
    def line_width(self) -> float:
        return self._line_width
