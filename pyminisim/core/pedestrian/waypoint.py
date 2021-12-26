from typing import Tuple, Optional
import numpy as np


class WaypointTracker:

    def __init__(self,
                 world_size: Tuple[float, float],
                 min_sample_distance=3.0,
                 max_sample_trials=100,
                 reach_distance=0.3):
        assert len(world_size) == 2
        self._world_size = world_size
        self._min_sample_distance = min_sample_distance
        self._max_sample_trials = max_sample_trials
        self._reach_distance = reach_distance
        self._waypoints = None

    @property
    def current_waypoints(self) -> Optional[np.ndarray]:
        return self._waypoints.copy() if self._waypoints is not None else None

    def sample_waypoints(self, agents_positions: np.ndarray) -> np.ndarray:
        assert agents_positions.shape[1] == 2
        self._waypoints = np.stack([self._sample_single_waypoint(p) for p in agents_positions])
        return self._waypoints.copy()

    def update_waypoints(self, agents_positions: np.ndarray) -> np.ndarray:
        if self._waypoints is None:
            raise RuntimeError("Waypoints are not initialized")
        assert agents_positions.shape == self._waypoints.shape
        self._waypoints = np.stack([
            self._sample_single_waypoint(position) if not self._waypoint_reached(waypoint, position) else waypoint
            for waypoint, position in zip(self._waypoints, agents_positions)])
        return self._waypoints.copy()

    def _sample_single_waypoint(self, agent_position: np.ndarray):
        for _ in range(self._max_sample_trials):
            sampled_point = np.random.uniform(low=np.array([0.0, 0.0]),
                                              high=np.array([self._world_size[0], self._world_size[1]]))
            if np.linalg.norm(agent_position - sampled_point) < self._min_sample_distance:
                continue
            print("Sampled point: ", sampled_point)
            return sampled_point
        raise RuntimeError("Failed to sample waypoint")

    def _waypoint_reached(self,
                          waypoint: np.ndarray,
                          agent_position: np.ndarray) -> bool:
        return np.linalg.norm(waypoint - agent_position) <= self._reach_distance
