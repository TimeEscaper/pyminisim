from typing import Union

import numpy as np

from pyminisim.core import AbstractWorldMapState, AbstractStaticWorldMap


class EmptyWorldMapState(AbstractWorldMapState):

    def __init__(self):
        super(EmptyWorldMapState, self).__init__()


class EmptyWorld(AbstractStaticWorldMap):

    def __init__(self):
        super(EmptyWorld, self).__init__(EmptyWorldMapState())

    def closest_distance_to_obstacle(self, point: np.ndarray) -> Union[float, np.ndarray]:
        assert point.shape == (2,) or (len(point.shape) == 2 and point.shape[1] == 2)
        if point.shape == (2,):
            return np.inf
        return np.repeat(np.inf, point.shape[0])

    def is_occupied(self, point: np.ndarray) -> Union[bool, np.ndarray]:
        if len(point.shape) == 2:
            return np.array([False for _ in range(point.shape[0])])
        return False

    def is_collision(self, agent_points: np.ndarray, agent_radii: Union[float, np.ndarray],
                     eps_margin: float = 0.05) -> Union[bool, np.ndarray]:
        if len(agent_points.shape) == 1:
            return False
        if len(agent_points.shape) == 2:
            return np.zeros((agent_points.shape[0]), dtype=bool)
        raise ValueError(f"Unsupported agent_point shape: {agent_points.shape}")
