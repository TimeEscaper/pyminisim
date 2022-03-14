from abc import ABC, abstractmethod
from typing import Optional

import numpy as np


class AbstractRobotMotionModelState(ABC):

    def __init__(self,
                 pose: np.ndarray,
                 velocity: np.ndarray,
                 control: np.ndarray):
        assert pose.shape == (3,)
        assert velocity.shape == (3,)
        self._pose = pose
        self._velocity = velocity
        self._control = control

    @property
    def pose(self) -> np.ndarray:
        return self._pose

    @property
    def velocity(self) -> np.ndarray:
        return self._velocity

    @property
    def control(self) -> np.ndarray:
        return self._control


class AbstractRobotMotionModel(ABC):

    def __init__(self,
                 initial_pose: np.ndarray,
                 initial_velocity: np.ndarray,
                 initial_control: np.ndarray):
        assert initial_pose.shape == (3,)
        assert initial_velocity.shape == (3,)
        self._state = self._init_state(initial_pose, initial_velocity, initial_control)

    def _init_state(self,
                    initial_pose: np.ndarray,
                    initial_velocity: np.ndarray,
                    initial_control: np.ndarray) -> AbstractRobotMotionModelState:
        raise NotImplementedError()

    @property
    def state(self) -> AbstractRobotMotionModelState:
        return self._state

    def reset(self,
              initial_pose: np.ndarray,
              initial_velocity: np.ndarray,
              initial_control: np.ndarray):
        assert initial_pose.shape == (3,)
        assert initial_velocity.shape == (3,)
        self._state = self._init_state(initial_pose, initial_velocity, initial_control)

    def reset_to_state(self, state: AbstractRobotMotionModelState):
        self._state = state

    @abstractmethod
    def step(self, dt: float, control: Optional[np.ndarray] = None):
        raise NotImplementedError()
