from abc import ABC, abstractmethod

from pyminisim.core import SimulationState


class AbstractSensorSkin(ABC):

    @abstractmethod
    def render(self, screen, sim_state: SimulationState):
        raise NotImplementedError()
