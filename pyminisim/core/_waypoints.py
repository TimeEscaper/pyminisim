from abc import ABC, abstractmethod
from typing import Optional

import numpy as np


class AbstractWaypointTrackerState(ABC):

    def __init__(self, current_waypoints: np.ndarray):
        self._current_waypoints = current_waypoints.copy()

    @property
    def current_waypoints(self) -> np.ndarray:
        return self._current_waypoints


class AbstractWaypointTracker(ABC):

    def __init__(self, state: AbstractWaypointTrackerState):
        self._state = state

    @property
    def state(self) -> AbstractWaypointTrackerState:
        return self._state

    @abstractmethod
    def resample_all(self, agents_poses: np.ndarray) -> np.ndarray:
        raise NotImplementedError()

    @abstractmethod
    def update_waypoints(self, agents_poses: np.ndarray) -> np.ndarray:
        raise NotImplementedError()

    @abstractmethod
    def set_waypoints(self, waypoints: np.ndarray):
        raise NotImplementedError()

    def reset_to_state(self, state: AbstractWaypointTrackerState):
        self._state = state
