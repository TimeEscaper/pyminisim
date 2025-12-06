import numpy as np

from abc import ABC, abstractmethod
from typing import Optional
from pyminisim.core._world_map import AbstractWorldMap
from pyminisim.core._motion_state import AbstractRobotMotionModelState
from pyminisim.core._constants import DEFAULT_ROBOT_RADIUS


class AbstractRobotMotionModel(ABC):

    def __init__(self,
                 initial_state: AbstractRobotMotionModelState,
                 radius: float = DEFAULT_ROBOT_RADIUS):
        assert radius > 0.
        self._state = initial_state
        self._radius = radius

    @property
    def state(self) -> AbstractRobotMotionModelState:
        return self._state

    @property
    def radius(self) -> float:
        return self._radius

    def reset_to_state(self, state: AbstractRobotMotionModelState):
        self._state = state

    @abstractmethod
    def step(self, dt: float, world_map: AbstractWorldMap, control: Optional[np.ndarray] = None) -> bool:
        raise NotImplementedError()
