from abc import ABC, abstractmethod

from ._world_state import WorldState
from ._world_map import AbstractWorldMap


class AbstractSensorConfig(ABC):
    pass


class AbstractSensorReading(ABC):
    pass


class AbstractSensor(ABC):

    def __init__(self, name: str):
        self._name = name

    @property
    def sensor_name(self) -> str:
        return self._name

    @property
    @abstractmethod
    def sensor_config(self) -> AbstractSensorConfig:
        raise NotImplementedError()

    @abstractmethod
    def get_reading(self, world_state: WorldState, world_map: AbstractWorldMap) -> AbstractSensorReading:
        raise NotImplementedError()
