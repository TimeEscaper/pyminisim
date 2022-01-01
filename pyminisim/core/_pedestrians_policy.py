from abc import ABC, abstractmethod

import numpy as np


class AbstractPedestriansPolicy(ABC):

    def __init__(self,
                 initial_poses: np.ndarray,
                 initial_velocities: np.ndarray):
        self._poses = initial_poses
        self._velocities = initial_velocities

    def reset(self,
              initial_poses: np.ndarray,
              initial_velocities: np.ndarray):
        self._poses = initial_poses
        self._velocities = initial_velocities

    @property
    def poses(self) -> np.ndarray:
        return self._poses

    @property
    def velocities(self) -> np.ndarray:
        return self._velocities

    @abstractmethod
    def step(self, dt: float, robot_pose: np.ndarray, robot_velocity: np.ndarray):
        raise NotImplementedError()
