from abc import ABC, abstractmethod
from dataclasses import dataclass

from ._world_state import WorldState
from ._world_map import AbstractWorldMap


class AbstractSensorConfig(ABC):
    pass


class AbstractSensorReading(ABC):
    pass


class AbstractSensor(ABC):

    def __init__(self, name: str, period: float):
        assert period >= 0.
        self._name = name
        self._period = period

    @property
    def sensor_name(self) -> str:
        return self._name

    @property
    def period(self) -> float:
        return self._period

    @property
    @abstractmethod
    def sensor_config(self) -> AbstractSensorConfig:
        raise NotImplementedError()

    @abstractmethod
    def get_reading(self, world_state: WorldState, world_map: AbstractWorldMap) -> AbstractSensorReading:
        raise NotImplementedError()


@dataclass
class SensorState:
    reading: AbstractSensorReading
    hold_time: float
