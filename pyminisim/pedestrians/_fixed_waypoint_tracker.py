from typing import Tuple, Optional

import numpy as np

from pyminisim.core import AbstractWaypointTrackerState, AbstractWaypointTracker


class FixedWaypointTrackerState(AbstractWaypointTrackerState):

    def __init__(self,
                 current_waypoints: np.ndarray,
                 current_indices: Optional[np.ndarray] = None):
        assert len(current_waypoints.shape) == 2 and current_waypoints.shape[1] == 2
        n_agents = current_waypoints.shape[0]
        if current_indices is None:
            current_indices = np.zeros((n_agents,)).astype(np.int)

        super(FixedWaypointTrackerState, self).__init__(current_waypoints)
        self._current_indices = current_indices

    @property
    def current_indices(self) -> np.ndarray:
        return self._current_indices


class FixedWaypointTracker(AbstractWaypointTracker):

    def __init__(self,
                 waypoints: np.ndarray,
                 reach_distance=0.3):
        super(FixedWaypointTracker, self).__init__()
        self._reach_distance = reach_distance
        self._waypoints = waypoints.copy()
        self._state = FixedWaypointTrackerState(waypoints[:, 0, :])

    @property
    def state(self) -> FixedWaypointTrackerState:
        return self._state

    def resample_all(self, agents_poses: np.ndarray) -> np.ndarray:
        assert agents_poses.shape[1] == 3
        return self._state.current_waypoints

    def update_waypoints(self, agents_poses: np.ndarray) -> np.ndarray:
        assert agents_poses.shape[0] == self._waypoints.shape[0]

        current_indices = self._state.current_indices
        current_waypoints = self._state.current_waypoints
        n_points = self._waypoints.shape[1]
        new_waypoints = np.zeros_like(current_waypoints)
        # TODO: Vectorization
        for i in range(agents_poses.shape[0]):
            if self._waypoint_reached(current_waypoints[i], agents_poses[i, :2]):
                current_indices[i] = min(current_indices[i] + 1, n_points - 1)
            new_waypoints[i, :] = self._waypoints[i, current_indices[i], :]
        self._state = FixedWaypointTrackerState(new_waypoints, current_indices)

        return self._state.current_waypoints.copy()

    def reset_to_state(self, state: FixedWaypointTrackerState):
        self._state = state

    def sample_independent_points(self, n_points: int, min_cross_distance: Optional[float] = None):
        pass

    def _waypoint_reached(self,
                          waypoint: np.ndarray,
                          agent_position: np.ndarray) -> bool:
        return np.linalg.norm(waypoint - agent_position) <= self._reach_distance
