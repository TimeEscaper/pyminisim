from abc import ABC, abstractmethod

import numpy as np


class AbstractRobotMotionModel(ABC):

    def __init__(self,
                 initial_pose: np.ndarray,
                 initial_velocity: np.ndarray,
                 initial_control: np.ndarray):
        assert initial_pose.shape == (3,)
        assert initial_velocity.shape == (3,)
        self._pose = initial_pose
        self._velocity = initial_velocity
        self._control = initial_control

    def reset(self,
              initial_pose: np.ndarray,
              initial_velocity: np.ndarray,
              initial_control: np.ndarray):
        assert initial_pose.shape == (3,)
        assert initial_velocity.shape == (3,)
        self._pose = initial_pose
        self._velocity = initial_velocity
        self._control = initial_control

    @property
    def pose(self) -> np.ndarray:
        return self._pose

    @property
    def velocity(self) -> np.ndarray:
        return self._velocity

    @property
    def control(self) -> np.ndarray:
        return self._control

    @control.setter
    def control(self, value: np.ndarray):
        self._control = value

    @abstractmethod
    def step(self, dt: float):
        raise NotImplementedError()
