from abc import ABC, abstractmethod
from typing import Optional

import numpy as np

from ._waypoints import AbstractWaypointTracker


class AbstractPedestriansPolicy(ABC):

    def __init__(self,
                 initial_poses: np.ndarray,
                 initial_velocities: np.ndarray,
                 waypoint_tracker: AbstractWaypointTracker):
        self._poses = initial_poses
        self._velocities = initial_velocities
        self._waypoint_tracker = waypoint_tracker

        if self._waypoint_tracker.current_waypoints is None:
            self._waypoint_tracker.resample_all(initial_poses)

    def reset(self,
              initial_poses: np.ndarray,
              initial_velocities: np.ndarray,
              initial_waypoints: Optional[np.ndarray]):
        self._poses = initial_poses
        self._velocities = initial_velocities
        if initial_waypoints is None:
            self._waypoint_tracker.resample_all(self._poses)
        else:
            self._waypoint_tracker.set_waypoints(initial_waypoints)

    @property
    def poses(self) -> np.ndarray:
        return self._poses

    @property
    def velocities(self) -> np.ndarray:
        return self._velocities

    @abstractmethod
    def step(self, dt: float, robot_pose: Optional[np.ndarray], robot_velocity: Optional[np.ndarray]):
        raise NotImplementedError()
