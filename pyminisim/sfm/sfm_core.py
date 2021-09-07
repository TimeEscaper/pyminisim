import numpy as np
from typing import List, Union

from pyminisim.objects.agents_array import SFMAgentsArray
from pyminisim.objects.obstacle import SimObstacle
from pyminisim.measures.point import Point


class SFMCore:

    def __init__(self,
                 agents: SFMAgentsArray,
                 obstacles: List[SimObstacle],
                 obstacle_force_sigma: float = 0.2,
                 obstacle_force_max_distance: float = 0.2):
        self._agents = agents
        self._obstacles = obstacles
        self._obstacle_force_sigma = obstacle_force_sigma
        self._obstacle_force_max_distance = obstacle_force_max_distance

    def _calculate_obstacle_force(self) -> np.ndarray:
        forces = np.zeros((self._agents.states.shape[0], 2))
        for i in range(self._agents.states.shape[0]):
            agent_point = self._agents.states[i][:2]
            closest_obstacle_point = self._find_closest_obstacle_point(Point(agent_point))
            diff = closest_obstacle_point.vector - agent_point
            distance = np.linalg.norm(diff) - self._agents.parameters[i, 0]
            if distance >= self._obstacle_force_max_distance:
                continue
            force = (diff / np.linalg.norm(diff)) * np.exp(-distance / self._obstacle_force_sigma)
            forces[i] = force
        return forces

    def _find_closest_obstacle_point(self, point: Point) -> Point:
        closest_points = [obs.shape.get_closest_point(point) for obs in self._obstacles]
        distances = [closest_point.distance(point) for closest_point in closest_points]
        return closest_points[np.argmin(distances)]
        # if min_distance >= self._obstacle_force_max_distance:
        #     return None
        # else:
        #     return min_distance
