from abc import ABC

from pyminisim.measures.point import Point
from pyminisim.measures.velocity import Velocity


class SFMAgent(ABC):
    def __init__(self,
                 agent_id: int,
                 # initial_position: Point,
                 # initial_velocity: Velocity,
                 # max_velocity: float,
                 # radius: float,
                 name: str = ""):
        self._agent_id = agent_id
        self._name = name
        # self.position = initial_position
        # self.velocity = initial_velocity
        # self._max_vel = max_velocity
        # self._radius = radius

    @property
    def agent_id(self) -> int:
        return self._agent_id

    @property
    def agent_name(self) -> str:
        return self._name

    # @property
    # def position(self) -> Point:
    #     return self._position
    #
    # @position.setter
    # def position(self, value: Point):
    #     self._position = value
    #
    # @property
    # def velocity(self) -> Velocity:
    #     return self._velocity
    #
    # @velocity.setter
    # def velocity(self, value: Velocity):
    #     self._velocity = value
    #
    # @property
    # def max_velocity(self) -> float:
    #     return self._max_vel
    #
    # @property
    # def radius(self) -> float:
    #     return self._radius
