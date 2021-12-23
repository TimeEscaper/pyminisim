from typing import Tuple
import numpy as np


class WaypointSampler:

    def __init__(self,
                 world_size: Tuple[float, float],
                 min_sample_distance=3.0,
                 max_sample_trials=100):
        assert len(world_size) == 2
        self._world_size = world_size
        self._min_sample_distance = min_sample_distance
        self._max_sample_trials = max_sample_trials

    def sample_waypoint(self,
                        agent_position: np.ndarray) -> np.ndarray:
        assert agent_position.shape[0] == 2
        for _ in range(self._max_sample_trials):
            sampled_point = np.random.uniform(low=np.array([0.0, 0.0]),
                                              high=np.array([self._world_size[0], self._world_size[1]]))
            if np.linalg.norm(agent_position - sampled_point) < self._min_sample_distance:
                continue
            return sampled_point
        raise RuntimeError("Failed to sample waypoint")
