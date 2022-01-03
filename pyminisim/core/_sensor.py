from abc import ABC, abstractmethod

from ._world_state import WorldState


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
    def get_reading(self, world_state: WorldState) -> AbstractSensorReading:
        raise NotImplementedError()
