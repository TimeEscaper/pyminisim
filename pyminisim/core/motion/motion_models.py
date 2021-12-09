from typing import Optional

import numpy as np

from pyminisim.core.util import wrap_angle


class UnicycleMotion:
    def __init__(self, initial_poses: np.ndarray, initial_velocities: Optional[np.ndarray] = None):
        assert len(initial_poses.shape) == 2 and initial_poses.shape[1] == 3
        self._poses = initial_poses.copy()
        if initial_velocities is not None:
            self._velocities = initial_velocities
        else:
            self._velocities = np.zeros((self._poses.shape[0], 2))

    @property
    def velocities(self) -> np.ndarray:
        return self._velocities

    @velocities.setter
    def velocities(self, value: np.ndarray):
        assert value.shape == (self._poses.shape, 2)
        self._velocities = value

    @property
    def poses(self) -> np.ndarray:
        return self._poses

    def step(self, dt: float):
        x = self._poses[:, 0] + self._velocities[:, 0] * np.cos(np.deg2rad(self._poses[:, 2])) * dt
        y = self._poses[:, 1] + self._velocities[:, 0] * np.sin(np.deg2rad(self._poses[:, 2])) * dt
        w = wrap_angle(self._poses[:, 2] + self._velocities[:, 1] * dt)
        self._poses = np.stack((x, y, w), axis=1)
