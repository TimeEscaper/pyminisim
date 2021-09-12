from typing import List

from pyminisim.objects.agent import SFMAgent
from pyminisim.objects.agent_state import AgentState
from pyminisim.sfm.sfm_core import SFMCore
from pyminisim.measures.point import Point
from pyminisim.measures.velocity import Velocity
from pyminisim.measures.trajectory import Trajectory


class Simulation:

    def __init__(self):
        self._agents = [SFMAgent(agent_id=1)]
        self._agents_states = [AgentState(1, position=Point([0, 0]), velocity=Velocity([0.3, 0.3]))]
        trajectory = Trajectory([Point([1, 1]), Point([2, 2]), Point([3, 3])])
        self._agents_states[0].trajectory = trajectory

        self._sfm = SFMCore()
        self._sfm.setup(initial_states=self._agents_states, obstacles=[])

    def step(self):
        self._sfm.step()
        new_state = self._sfm.get_states()
        self._agents_states[0].position = new_state[0][0]
        self._agents_states[0].velocity = new_state[0][1]
        # if self._agents_states[0].position.distance(self._agents_states[0].trajectory.current_waypoint) < 0.15:
        #     self._agents_states[0].trajectory.

    def render(self):
        pass
