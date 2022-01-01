from abc import ABC, abstractmethod
from typing import Dict

from pyminisim.core import SimulationState


class AbstractSensor(ABC):

    def __init__(self, name: str):
        self._name = name

    @property
    def sensor_name(self) -> str:
        return self._name

    @abstractmethod
    def get_reading(self, simulation_state: SimulationState) -> Dict:
        raise NotImplementedError()
