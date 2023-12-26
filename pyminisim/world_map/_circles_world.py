from typing import Union

import numpy as np
from scipy.spatial.distance import cdist

from pyminisim.core import AbstractWorldMapState, AbstractStaticWorldMap


class _CirclesWorldState(AbstractWorldMapState):

    def __init__(self, circles: np.ndarray):
        assert len(circles.shape) == 2 and circles.shape[1] == 3
        super(_CirclesWorldState, self).__init__()
        self._circles = circles

    @property
    def circles(self) -> np.ndarray:
        return self._circles


class CirclesWorld(AbstractStaticWorldMap):

    def __init__(self, circles: np.ndarray):
        """
        :param circles: List of circles in format [[x, y, radius]] (in metres)
        """
        super(CirclesWorld, self).__init__(_CirclesWorldState(circles.copy()))

    def closest_distance_to_obstacle(self, point: np.ndarray) -> \
            Union[float, np.ndarray]:
        assert point.shape == (2,) or (len(point.shape) == 2 and point.shape[1] == 2)

        if len(point.shape) == 1:
            return np.min(np.linalg.norm(point - self._state.circles[:, :2], axis=1) - self._state.circles[:, 2])

        pairwise_dist = cdist(point, self._state.circles[:, :2], metric="euclidean")
        pairwise_dist = pairwise_dist - self._state.circles[:, 2]

        return np.min(pairwise_dist, axis=1)

    def is_occupied(self, point: np.ndarray) -> Union[bool, np.ndarray]:
        assert point.shape == (2,) or (len(point.shape) == 2 and point.shape[1] == 2)

        if len(point.shape) == 1:
            return (np.linalg.norm(point - self._state.circles[:, :2], axis=1) - self._state.circles[:, 2] > 0.).all()

        pairwise_dist = cdist(point, self._state.circles[:, :2], metric="euclidean")
        pairwise_dist = pairwise_dist - self._state.circles[:, 2]
        result = (pairwise_dist <= 0.).any(axis=1)
        return result

    @property
    def circles(self) -> np.ndarray:
        return self._state.circles.copy()
