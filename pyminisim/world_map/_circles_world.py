from typing import Union

import numpy as np

from pyminisim.core import AbstractWorldMap


class CirclesWorld(AbstractWorldMap):

    def __init__(self, circles: np.ndarray):
        """
        :param circles: List of circles in format [[x, y, radius]] (in metres)
        """
        assert len(circles.shape) == 2 and circles.shape[1] == 3
        super(CirclesWorld, self).__init__()
        self._circles = circles.copy()

    def closest_distance_to_obstacle(self, point: np.ndarray) -> Union[float, np.ndarray]:
        assert point.shape == (2,) or (len(point.shape) == 2 and point.shape[1] == 2)
        if point.shape == (2,):
            distances = np.linalg.norm(point - self._circles[:, :2]).flatten()
            min_idx = np.argmin(distances)
            return distances[min_idx] - self._circles[min_idx, 2]
        else:
            # TODO: Vectorization
            return np.array([self.closest_distance_to_obstacle(p) for p in point])

    @property
    def circles(self) -> np.ndarray:
        return self._circles.copy()
