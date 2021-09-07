import numpy as np
from typing import List

from pyminisim.objects.agent import SFMAgent


class SFMAgentsArray:

    def __init__(self, agents: List[SFMAgent]):
        agents_states = [[agent.position.x,
                          agent.position.y,
                          agent.velocity.x,
                          agent.velocity.y] for agent in agents]
        agents_params = [[agent.radius,
                          agent.max_velocity] for agent in agents]
        self._agents_states = np.array(agents_states)
        self._agents_params = np.array(agents_params)

    @property
    def states(self) -> np.ndarray:
        return self._agents_states

    @states.setter
    def states(self, value: np.ndarray):
        assert value.shape[1] == 4
        assert value.shape[0] == self._agents_states.shape[0]
        self._agents_states = value

    @property
    def parameters(self) -> np.ndarray:
        return self._agents_params
