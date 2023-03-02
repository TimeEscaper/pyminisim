from abc import ABC, abstractmethod
from typing import Optional, Tuple, List, Dict

import numpy as np


class AbstractWaypointTrackerState(ABC):

    def __init__(self, current_waypoints: Dict[int, np.ndarray]):
        self._current_waypoints = {}
        for k, v in current_waypoints.items():
            assert v.shape == (2,), f"Waypoints must be array of shape (2,), the {v.shape} array was passed for pedestrian {k}"
            self._current_waypoints[k] = v.copy()

    @property
    def current_waypoints(self) -> Dict[int, np.ndarray]:
        return self._current_waypoints


class AbstractWaypointTracker(ABC):

    @property
    def state(self) -> AbstractWaypointTrackerState:
        raise NotImplementedError()

    @abstractmethod
    def resample_all(self, agents_poses: Dict[int, np.ndarray]) -> Dict[int, np.ndarray]:
        raise NotImplementedError()

    @abstractmethod
    def update_waypoints(self, agents_poses: Dict[int, np.ndarray]) -> Dict[int, Tuple[np.ndarray, bool]]:
        raise NotImplementedError()

    @abstractmethod
    def reset_to_state(self, state: AbstractWaypointTrackerState):
        raise NotImplementedError()

    @abstractmethod
    def sample_independent_points(self, n_points: int, min_cross_distance: Optional[float] = None):
        raise NotImplementedError()
