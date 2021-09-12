import numpy as np
from typing import List, Union, Dict, Tuple
from pathlib import Path
import pysocialforce as psf

from pyminisim.objects.agents_array import SFMAgentsArray
from pyminisim.objects.agent import SFMAgent
from pyminisim.objects.obstacle import SimObstacle
from pyminisim.measures.point import Point
from pyminisim.measures.velocity import Velocity
from pyminisim.objects.agent_state import AgentState
from pyminisim.measures.shape import PolygonalShape


class SFMCore:

    def __init__(self): # psf_config_file: Path):
        # self._config = psf_config_file
        self._psf_sim: Union[psf.Simulator, None] = None
        self._id_to_idx: Union[Dict, None] = None
        self._idx_to_id: Union[Dict, None] = None

    def setup(self, initial_states: List[AgentState], obstacles: List[SimObstacle]):
        psf_state = np.array([[st.position.x, st.position.y,
                               st.velocity.x, st.velocity.y,
                               st.trajectory.current_waypoint.x,
                               st.trajectory.current_waypoint.y] for st in initial_states])
        indices = np.arange(0, len(initial_states))
        self._id_to_idx = {initial_states[i].agent_id: indices[i] for i in range(len(initial_states))}
        self._idx_to_id = {indices[i]: initial_states[i].agent_id for i in range(len(initial_states))}
        if len(obstacles) != 0:
            psf_obstacles = np.concatenate([obs.shape.get_segments() for obs in obstacles if obs.shape is PolygonalShape],
                                           axis=1)
        else:
            psf_obstacles = None
        self._psf_sim = psf.Simulator(psf_state, obstacles=psf_obstacles, config_file=None)  # TODO: Fix config

    def step(self):
        if self._psf_sim is not None:
            self._psf_sim.step()

    def get_states(self) -> Union[List[Tuple[Point, Velocity]], None]:
        if self._psf_sim is None:
            return None
        states = self._psf_sim.get_states()[0]
        return [(Point(states[i, 0:2]), Velocity(states[i, 2:4])) for i in range(states.shape[0])]


# class SFMCore:
#
#     def __init__(self,
#                  agents: SFMAgentsArray,
#                  obstacles: List[SimObstacle],
#                  obstacle_force_sigma: float = 0.2,
#                  obstacle_force_max_distance: float = 0.2):
#         self._agents = agents
#         self._obstacles = obstacles
#         self._obstacle_force_sigma = obstacle_force_sigma
#         self._obstacle_force_max_distance = obstacle_force_max_distance
#
#     def _calculate_obstacle_force(self) -> np.ndarray:
#         forces = np.zeros((self._agents.states.shape[0], 2))
#         for i in range(self._agents.states.shape[0]):
#             agent_point = self._agents.states[i][:2]
#             closest_obstacle_point = self._find_closest_obstacle_point(Point(agent_point))
#             diff = closest_obstacle_point.vector - agent_point
#             distance = np.linalg.norm(diff) - self._agents.parameters[i, 0]
#             if distance >= self._obstacle_force_max_distance:
#                 continue
#             force = (diff / np.linalg.norm(diff)) * np.exp(-distance / self._obstacle_force_sigma)
#             forces[i] = force
#         return forces
#
#     def _find_closest_obstacle_point(self, point: Point) -> Point:
#         closest_points = [obs.shape.get_closest_point(point) for obs in self._obstacles]
#         distances = [closest_point.distance(point) for closest_point in closest_points]
#         return closest_points[np.argmin(distances)]
#         # if min_distance >= self._obstacle_force_max_distance:
#         #     return None
#         # else:
#         #     return min_distance
