from typing import Union

from pyminisim.measures.point import Point
from pyminisim.measures.velocity import Velocity
from pyminisim.measures.trajectory import Trajectory


class AgentState:

    def __init__(self,
                 agent_id: int,
                 position: Union[Point, None] = None,
                 velocity: Union[Velocity, None] = None,
                 trajectory: Union[Trajectory, None] = None):
        self._id = agent_id
        self._position = position
        self._velocity = velocity
        self._trajectory = trajectory

    @property
    def agent_id(self) -> int:
        return self._id

    @property
    def position(self) -> Union[Point, None]:
        return self._position

    @position.setter
    def position(self, value: Point):
        self._position = value

    @property
    def velocity(self) -> Union[Velocity, None]:
        return self._velocity

    @velocity.setter
    def velocity(self, value: Velocity):
        self._velocity = value

    @property
    def trajectory(self) -> Union[Trajectory, None]:
        return self._trajectory

    @trajectory.setter
    def trajectory(self, value: Trajectory):
        self._trajectory = value
