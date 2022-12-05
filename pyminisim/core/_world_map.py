from abc import ABC, abstractmethod
from typing import Union

import numpy as np


class AbstractWorldMapState(ABC):
    pass


class AbstractWorldMap(ABC):

    def __init__(self, initial_state: AbstractWorldMapState):
        self._state = initial_state

    @property
    def current_state(self) -> AbstractWorldMapState:
        return self._state

    def reset_to_state(self, state: AbstractWorldMapState):
        self._state = state

    @abstractmethod
    def step(self, dt: float) -> None:
        raise NotImplementedError()

    @abstractmethod
    def closest_distance_to_obstacle(self, point: np.ndarray) -> Union[float, np.ndarray]:
        raise NotImplementedError()

    @abstractmethod
    def is_occupied(self, point: np.ndarray) -> Union[bool, np.ndarray]:
        raise NotImplementedError()


class AbstractStaticWorldMap(AbstractWorldMap, ABC):

    def __init__(self, initial_state: AbstractWorldMapState):
        super(AbstractStaticWorldMap, self).__init__(initial_state)

    def step(self, dt: float) -> None:
        return
