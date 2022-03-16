from abc import ABC, abstractmethod

import numpy as np


class AbstractWaypointTrackerState(ABC):

    def __init__(self, current_waypoints: np.ndarray):
        self._current_waypoints = current_waypoints

    @property
    def current_waypoints(self) -> np.ndarray:
        return self._current_waypoints


class AbstractWaypointTracker(ABC):

    @property
    def state(self) -> AbstractWaypointTrackerState:
        raise NotImplementedError()

    @abstractmethod
    def resample_all(self, agents_poses: np.ndarray) -> np.ndarray:
        raise NotImplementedError()

    @abstractmethod
    def update_waypoints(self, agents_poses: np.ndarray) -> np.ndarray:
        raise NotImplementedError()

    @abstractmethod
    def reset_to_state(self, state: AbstractWaypointTrackerState):
        raise NotImplementedError()
