from typing import Tuple, Optional

import numpy as np

from pyminisim.core import AbstractWaypointTrackerState, AbstractWaypointTracker


class FixedWaypointTrackerState(AbstractWaypointTrackerState):

    def __init__(self,
                 waypoints: np.ndarray,
                 current_indices: Optional[np.ndarray] = None):
        assert len(waypoints.shape) == 3 and waypoints.shape[2] == 2
        n_agents = waypoints.shape[0]
        n_points = waypoints.shape[1]
        if current_indices is not None:
            assert (current_indices < n_points).all()
        else:
            current_indices = np.zeros((n_agents,))
        current_indices = current_indices.astype(np.int)

        super(FixedWaypointTrackerState, self).__init__(waypoints[:, current_indices, :])
        self._waypoints = waypoints.copy()
        self._current_indices = current_indices

    @property
    def waypoints(self) -> np.ndarray:
        return self._waypoints.copy()

    @property
    def current_indices(self) -> np.ndarray:
        return self._current_indices.copy()

    @property
    def n_agents(self) -> int:
        return self._waypoints.shape[0]

    @property
    def n_points(self) -> int:
        return self._waypoints.shape[1]

    # @property
    # def next_waypoints(self) -> Optional[np.ndarray]:
    #     if self._next_waypoints is not None:
    #         return self._next_waypoints.copy()
    #     return None


class FixedWaypointTracker(AbstractWaypointTracker):

    def __init__(self,
                 waypoints: np.ndarray,
                 reach_distance=0.3):
        super(FixedWaypointTracker, self).__init__()
        self._reach_distance = reach_distance
        self._state = FixedWaypointTrackerState(waypoints)

    @property
    def state(self) -> FixedWaypointTrackerState:
        return self._state

    def resample_all(self, agents_poses: np.ndarray) -> np.ndarray:
        assert agents_poses.shape[1] == 3
        return self._state.current_waypoints

    def update_waypoints(self, agents_poses: np.ndarray) -> np.ndarray:
        assert agents_poses.shape[0] == self._state.n_agents

        current_indices = self._state.current_indices
        current_waypoints = self._state.current_waypoints
        n_points = self._state.n_points
        # TODO: Vectorization
        for i in range(agents_poses.shape[0]):
            if self._waypoint_reached(current_waypoints[i], agents_poses[i, :2]):
                current_indices[i] = min(current_indices[i] + 1, n_points - 1)
        self._state = FixedWaypointTrackerState(self._state.waypoints, current_indices)

        return self._state.current_waypoints.copy()

    def reset_to_state(self, state: FixedWaypointTrackerState):
        self._state = state

    def sample_independent_points(self, n_points: int, min_cross_distance: Optional[float] = None):
        pass

    def _waypoint_reached(self,
                          waypoint: np.ndarray,
                          agent_position: np.ndarray) -> bool:
        return np.linalg.norm(waypoint - agent_position) <= self._reach_distance
