from typing import Tuple, Optional, List, Dict

import numpy as np

from pyminisim.core import AbstractWaypointTrackerState, AbstractWaypointTracker


class RandomWaypointTrackerState(AbstractWaypointTrackerState):

    def __init__(self,
                 current_waypoints: Dict[int, np.ndarray],
                 next_waypoints: Dict[int, np.ndarray]):
        super(RandomWaypointTrackerState, self).__init__(current_waypoints)
        self._next_waypoints = next_waypoints

    @property
    def next_waypoints(self) -> Dict[int, np.ndarray]:
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

    def resample_all(self, agents_poses: Dict[int, np.ndarray]) -> Dict[int, np.ndarray]:

        current_waypoints_dict = {}
        next_waypoints_dict = {}

        agents_poses_stack = np.stack([agents_poses[k] for k in sorted(agents_poses.keys())], axis=0)

        next_waypoints = np.zeros((agents_poses_stack.shape[0], self._n_next_waypoints + 1, 2))
        next_waypoints[:, 0, :] = np.stack([self._sample_single_waypoint(p, self._min_sample_distance)
                                            for p in agents_poses_stack[:, :2]])
        for i in range(1, self._n_next_waypoints + 1):
            next_waypoints[:, i, :] = np.stack([self._sample_single_waypoint(p, self._min_sample_distance)
                                                for p in next_waypoints[:, i - 1, :]])
        current_waypoints = next_waypoints[:, 0, :]
        next_waypoints = next_waypoints[:, 1:, :]

        for i, agent_id in enumerate(sorted(agents_poses.keys())):
            current_waypoints_dict[agent_id] = current_waypoints[i, :]
            next_waypoints_dict[agent_id] = next_waypoints[i, :, :]

        self._state = RandomWaypointTrackerState(current_waypoints_dict, next_waypoints_dict)
        return current_waypoints_dict

    def update_waypoints(self, agents_poses: Dict[int, np.ndarray]) -> Dict[int, Tuple[np.ndarray, bool]]:
        if self._state is None:
            raise RuntimeError("Waypoint tracker is not initialized")
        assert agents_poses.keys() == self._state.current_waypoints.keys()

        current_waypoints = self._state.current_waypoints
        next_waypoints = self._state.next_waypoints

        current_waypoints_new = {}
        next_waypoints_new = {}

        for i, agent_pose in agents_poses.items():
            if not self._waypoint_reached(current_waypoints[i], agent_pose[:2]):
                current_waypoints_new[i] = current_waypoints[i].copy()
                next_waypoints_new[i] = next_waypoints[i].copy()
            else:
                current_waypoints_new[i] = next_waypoints[i][0, :]
                next_waypoints_updated = np.zeros_like(next_waypoints[i])
                next_waypoints_updated[:-1, :] = next_waypoints[i][1:, :]
                next_waypoints_updated[-1, :] = self._sample_single_waypoint(
                    next_waypoints[i][-1, :], self._min_sample_distance)
                next_waypoints_new[i] = next_waypoints_updated

        self._state = RandomWaypointTrackerState(current_waypoints_new, next_waypoints_new)

        return {k: (v.copy(), False) for k, v in current_waypoints_new.items()}

    def reset_to_state(self, state: RandomWaypointTrackerState):
        self._state = state

    def sample_independent_points(self, n_points: int, min_cross_distance: Optional[float] = None):
        assert n_points > 0
        if min_cross_distance is None:
            min_cross_distance = -np.inf

        sampled_points = np.random.uniform(low=np.array([-self._world_size[0] / 2., -self._world_size[0] / 2]),
                                           high=np.array([self._world_size[0] / 2., self._world_size[0] / 2]))\
            .reshape(1, -1)
        for i in range(1, n_points):
            point = self._sample_single_waypoint(sampled_points, min_cross_distance)
            sampled_points = np.vstack([sampled_points, point.reshape(1, -1)])

        return sampled_points

    def _sample_single_waypoint(self, other_positions: np.ndarray, min_distance: float):
        for _ in range(self._max_sample_trials):
            sampled_point = np.random.uniform(low=np.array([-self._world_size[0] / 2., -self._world_size[0] / 2]),
                                              high=np.array([self._world_size[0] / 2., self._world_size[0] / 2]))
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

