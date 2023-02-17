from abc import ABC, abstractmethod
from typing import Optional

import numpy as np

from ._waypoints import AbstractWaypointTrackerState, AbstractWaypointTracker


class AbstractPedestriansModelState(ABC):

    def __init__(self,
                 poses: np.ndarray,
                 velocities: np.ndarray,
                 waypoints_state: AbstractWaypointTrackerState):
        assert len(poses.shape) == 2 and poses.shape[1] == 3, f"Given {len(poses.shape)=} and {poses.shape[1]=}"
        assert len(velocities.shape) == 2  and velocities.shape[1] == 3, f"Given {len(velocities.shape)=} and {velocities.shape[1]=}"
        assert poses.shape[0] == velocities.shape[0]
        self._poses = poses
        self._velocities = velocities
        self._waypoints = waypoints_state

    @property
    def poses(self) -> np.ndarray:
        return self._poses

    @property
    def velocities(self) -> np.ndarray:
        return self._velocities

    @property
    def waypoints(self) -> AbstractWaypointTrackerState:
        return self._waypoints


class AbstractPedestriansModel(ABC):

    # def __init__(self,
    #              initial_poses: np.ndarray,
    #              initial_velocities: np.ndarray,
    #              waypoint_tracker: AbstractWaypointTracker):
    #     self._poses = initial_poses
    #     self._velocities = initial_velocities
    #     self._waypoint_tracker = waypoint_tracker
    #
    #     if self._waypoint_tracker.current_waypoints is None:
    #         self._waypoint_tracker.resample_all(initial_poses)

    # def reset(self,
    #           initial_poses: np.ndarray,
    #           initial_velocities: np.ndarray,
    #           initial_waypoints: Optional[np.ndarray]):
    #     self._poses = initial_poses
    #     self._velocities = initial_velocities
    #     if initial_waypoints is None:
    #         self._waypoint_tracker.resample_all(self._poses)
    #     else:
    #         self._waypoint_tracker.set_waypoints(initial_waypoints)

    @property
    @abstractmethod
    def state(self) -> AbstractPedestriansModelState:
        raise NotImplementedError()

    @abstractmethod
    def step(self, dt: float, robot_pose: Optional[np.ndarray], robot_velocity: Optional[np.ndarray]):
        raise NotImplementedError()

    @abstractmethod
    def reset_to_state(self, state: AbstractPedestriansModelState):
        raise NotImplementedError()
