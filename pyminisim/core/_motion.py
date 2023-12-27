import numpy as np

from abc import ABC, abstractmethod
from typing import Optional
from pyminisim.core._world_map import AbstractWorldMap
from pyminisim.core._motion_state import AbstractRobotMotionModelState


class AbstractRobotMotionModel(ABC):

    def __init__(self,
                 initial_state: AbstractRobotMotionModelState):
        self._state = initial_state

    @property
    def state(self) -> AbstractRobotMotionModelState:
        return self._state

    def reset_to_state(self, state: AbstractRobotMotionModelState):
        self._state = state

    @abstractmethod
    def step(self, dt: float, world_map: AbstractWorldMap, control: Optional[np.ndarray] = None) -> bool:
        raise NotImplementedError()
