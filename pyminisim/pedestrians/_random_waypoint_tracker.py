from typing import Tuple, Optional

import numpy as np

from pyminisim.core import AbstractWaypointTrackerState, AbstractWaypointTracker


class RandomWaypointTrackerState(AbstractWaypointTrackerState):

    def __init__(self,
                 current_waypoints: np.ndarray,
                 next_waypoints: np.ndarray):
        assert len(current_waypoints.shape) == 2 and len(next_waypoints.shape) == 3
        assert current_waypoints.shape[1] == 2 and next_waypoints.shape[2] == 2
        assert current_waypoints.shape[0] == next_waypoints.shape[0]
        super(RandomWaypointTrackerState, self).__init__(current_waypoints)
        self._next_waypoints = next_waypoints

    @property
    def next_waypoints(self) -> np.ndarray:
        return self._next_waypoints


class RandomWaypointTracker(AbstractWaypointTracker):

    def __init__(self,
                 world_size: Tuple[float, float],
                 min_sample_distance=3.0,
                 max_sample_trials=100,
                 reach_distance=0.3,
                 n_next_waypoints: int = 10):
        assert len(world_size) == 2
        assert n_next_waypoints >= 1
        super(RandomWaypointTracker, self).__init__()
        self._world_size = world_size
        self._min_sample_distance = min_sample_distance
        self._max_sample_trials = max_sample_trials
        self._reach_distance = reach_distance
        self._n_next_waypoints = n_next_waypoints
        self._state: Optional[RandomWaypointTrackerState] = None

    @property
    def state(self) -> RandomWaypointTrackerState:
        return self._state

    def resample_all(self, agents_poses: np.ndarray) -> np.ndarray:
        assert agents_poses.shape[1] == 3
        next_waypoints = np.zeros((agents_poses.shape[0], self._n_next_waypoints + 1, 2))
        next_waypoints[:, 0, :] = np.stack([self._sample_single_waypoint(p, self._min_sample_distance)
                                            for p in agents_poses[:, :2]])
        for i in range(1, self._n_next_waypoints + 1):
            next_waypoints[:, i, :] = np.stack([self._sample_single_waypoint(p, self._min_sample_distance)
                                                for p in next_waypoints[:, i - 1, :]])
        current_waypoints = next_waypoints[:, 0, :]
        next_waypoints = next_waypoints[:, 1:, :]
        self._state = RandomWaypointTrackerState(current_waypoints, next_waypoints)
        return current_waypoints.copy()

    def update_waypoints(self, agents_poses: np.ndarray) -> np.ndarray:
        if self._state is None:
            raise RuntimeError("Waypoint tracker is not initialized")
        assert agents_poses.shape[0] == self._state.current_waypoints.shape[0]

        current_waypoints = self._state.current_waypoints.copy()
        next_waypoints = self._state.next_waypoints.copy()
        for i in range(agents_poses.shape[0]):
            if not self._waypoint_reached(current_waypoints[i], agents_poses[i, :2]):
                continue
            current_waypoints[i, :] = next_waypoints[i, 0, :].copy()
            new_waypoint = self._sample_single_waypoint(next_waypoints[i, -1, :], self._min_sample_distance)
            next_waypoints[i, :-1, :] = next_waypoints[i, 1:, :]
            next_waypoints[i, -1, :] = new_waypoint

        self._state = RandomWaypointTrackerState(current_waypoints, next_waypoints)

        return current_waypoints.copy()

    def reset_to_state(self, state: RandomWaypointTrackerState):
        self._state = state

    def sample_independent_points(self, n_points: int, min_cross_distance: Optional[float] = None):
        assert n_points > 0
        if min_cross_distance is None:
            min_cross_distance = -np.inf

        sampled_points = np.random.uniform(low=np.array([0.0, 0.0]),
                                           high=np.array([self._world_size[0], self._world_size[1]])).reshape(1, -1)
        for i in range(1, n_points):
            point = self._sample_single_waypoint(sampled_points, min_cross_distance)
            sampled_points = np.vstack([sampled_points, point.reshape(1, -1)])

        return sampled_points

    def _sample_single_waypoint(self, other_positions: np.ndarray, min_distance: float):
        for _ in range(self._max_sample_trials):
            sampled_point = np.random.uniform(low=np.array([0.0, 0.0]),
                                              high=np.array([self._world_size[0], self._world_size[1]]))
            if len(other_positions.shape) == 1:
                sampled_dist = np.linalg.norm(other_positions - sampled_point)
            elif len(other_positions.shape) == 2:
                sampled_dist = np.min(np.linalg.norm(other_positions - sampled_point, axis=1))
            else:
                raise RuntimeError("Unsupported agents positions shape")
            if sampled_dist < min_distance:
                continue
            return sampled_point
        raise RuntimeError("Failed to sample waypoint")

    def _waypoint_reached(self,
                          waypoint: np.ndarray,
                          agent_position: np.ndarray) -> bool:
        return np.linalg.norm(waypoint - agent_position) <= self._reach_distance

