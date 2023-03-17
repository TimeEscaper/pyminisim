from typing import Tuple, Optional, List, Dict, Union

import numpy as np

from pyminisim.core import AbstractWaypointTrackerState, AbstractWaypointTracker


class FixedWaypointTrackerState(AbstractWaypointTrackerState):

    def __init__(self,
                 current_waypoints: Dict[int, np.ndarray],
                 current_indices: Optional[Dict[int, int]] = None):
        if current_indices is None:
            current_indices = {k: 0 for k in current_waypoints.keys()}
        else:
            assert current_waypoints.keys() == current_indices.keys()

        super(FixedWaypointTrackerState, self).__init__(current_waypoints)
        self._current_indices = current_indices

    @property
    def current_indices(self) -> Dict[int, int]:
        return self._current_indices


class FixedWaypointTracker(AbstractWaypointTracker):

    def __init__(self,
                 waypoints: Union[Dict[int, np.ndarray], np.ndarray],
                 reach_distance=0.3):
        super(FixedWaypointTracker, self).__init__()
        self._reach_distance = reach_distance
        if isinstance(waypoints, np.ndarray):
            assert len(waypoints.shape) == 3 and waypoints.shape[2] == 2, \
                f"Waypoints must have shape of (n_pedestrians, n_points, 2), the {waypoints.shape} array is passed"
            self._all_waypoints = {i: waypoints[i] for i in range(waypoints.shape[0])}
        else:
            self._all_waypoints = {}
            for k, v in waypoints:
                assert len(v.shape) == 2 and v.shape[1] == 2, \
                    f"waypoints must have shape (n_points, 2), {v.shape} array is passed for pedestrian {k}"
                self._all_waypoints[k] = v.copy()

        self._current_indices = {k: 0 for k in self._all_waypoints.keys()}

        self._state = FixedWaypointTrackerState(self._get_current_waypoints(), self._current_indices)

    @property
    def state(self) -> FixedWaypointTrackerState:
        return self._state

    def resample_all(self, agents_poses: Dict[int, np.ndarray]) -> Dict[int, np.ndarray]:
        return self._get_current_waypoints()

    def update_waypoints(self, agents_poses: Dict[int, np.ndarray]) -> Dict[int, Tuple[np.ndarray, bool]]:
        assert agents_poses.keys() == self._all_waypoints.keys()

        result = {}
        for agent_id, agent_pose in agents_poses.items():
            current_index = self._current_indices[agent_id]
            if self._waypoint_reached(self._all_waypoints[agent_id][current_index], agent_pose[:2]):
                new_index = self._current_indices[agent_id] + 1
                is_steady = False
                if new_index >= self._all_waypoints[agent_id].shape[0]:
                    new_index = self._all_waypoints[agent_id].shape[0] - 1
                    is_steady = True
                new_waypoint = self._all_waypoints[agent_id][new_index]
                result[agent_id] = (new_waypoint, is_steady)

        return result

    def reset_to_state(self, state: FixedWaypointTrackerState):
        self._current_indices = state.current_indices

    def sample_independent_points(self, n_points: int, min_cross_distance: Optional[float] = None):
        pass

    def _waypoint_reached(self,
                          waypoint: np.ndarray,
                          agent_position: np.ndarray) -> bool:
        return np.linalg.norm(waypoint - agent_position) <= self._reach_distance

    def _get_current_waypoints(self) -> Dict[int, np.ndarray]:
        return {k: self._all_waypoints[k][i, :] for k, i in self._current_indices.items()}
