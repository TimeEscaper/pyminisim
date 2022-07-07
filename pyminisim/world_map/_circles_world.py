from typing import Union

import numpy as np
from scipy.spatial.distance import cdist

from pyminisim.core import AbstractWorldMap


class CirclesWorld(AbstractWorldMap):

    def __init__(self, circles: np.ndarray):
        """
        :param circles: List of circles in format [[x, y, radius]] (in metres)
        """
        assert len(circles.shape) == 2 and circles.shape[1] == 3
        super(CirclesWorld, self).__init__()
        self._circles = circles.copy()

    def closest_distance_to_obstacle(self, point: np.ndarray, radius: Union[float, np.ndarray] = 0.) -> \
            Union[float, np.ndarray]:
        assert point.shape == (2,) or (len(point.shape) == 2 and point.shape[1] == 2)
        if point.shape == (2,):
            return np.min(np.linalg.norm(point - self._circles[:, :2]).flatten() - self._circles[:, 2] - radius)
        else:
            # TODO: Vectorization
            return np.array([self.closest_distance_to_obstacle(p, r) for p, r in zip(point, radius)])

    def is_occupied(self, point: np.ndarray) -> Union[bool, np.ndarray]:
        if len(point.shape) == 1:
            return (np.linalg.norm(point - self._circles[:, :2]) - self._circles[:, 2] > 0.).all()

        pairwise_dist = cdist(point, self._circles[:, :2], metric="euclidean")
        pairwise_dist = pairwise_dist - self._circles[:, 2]
        result = (pairwise_dist > 0.).all(axis=1)
        return result

        # if len(point.shape) == 2:
        #     return np.array([self.is_occupied(p) for p in point])
        # for circle in self._circles:
        #     if np.linalg.norm(circle[:2] - point) <= circle[2]:
        #         return True
        # return False

    @property
    def circles(self) -> np.ndarray:
        return self._circles.copy()
