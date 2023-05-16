from abc import ABC, abstractmethod

import numpy as np

from pyminisim.core import SimulationState


class AbstractSensorSkin(ABC):

    @abstractmethod
    def render(self, screen, sim_state: SimulationState, global_offset: np.ndarray):
        raise NotImplementedError()


class AbstractMapSkin(ABC):

    @abstractmethod
    def render(self, screen, sim_state: SimulationState, global_offset: np.ndarray):
        raise NotImplementedError()
