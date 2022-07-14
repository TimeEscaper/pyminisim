from typing import Union

import numpy as np

from pyminisim.core import AbstractWorldMap


class EmptyWorld(AbstractWorldMap):

    def __init__(self):
        super(EmptyWorld, self).__init__()

    def closest_distance_to_obstacle(self, point: np.ndarray) -> Union[float, np.ndarray]:
        assert point.shape == (2,) or (len(point.shape) == 2 and point.shape[1] == 2)
        if point.shape == (2,):
            return np.inf
        return np.repeat(np.inf, point.shape[0])

    def is_occupied(self, point: np.ndarray) -> Union[bool, np.ndarray]:
        if len(point.shape) == 2:
            return np.array([False for _ in range(point.shape[0])])
        return False
