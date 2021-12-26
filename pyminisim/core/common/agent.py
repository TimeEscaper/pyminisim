from abc import ABC
from dataclasses import dataclass

import numpy as np

from pyminisim.core.common import Pose, Velocity


class AbstractAgent(ABC):

    def __init__(self, initial_pose: Pose, initial_velocity: Velocity):
        self._pose = initial_pose
        self._velocity = initial_velocity

    @property
    def pose(self) -> Pose:
        return self._pose

    # @pose.setter
    # def pose(self, value: Pose):
    #     self._pose = value

    @property
    def velocity(self) -> Velocity:
        return self._velocity

    # @velocity.setter
    # def velocity(self, value: Velocity):
    #     self._velocity = value


class PedestrianAgent(AbstractAgent):

    def __init__(self,
                 initial_pose: Pose,
                 initial_velocity: Velocity = Velocity(0.0, 0.0)):
        super(PedestrianAgent, self).__init__(initial_pose, initial_velocity)


class RobotAgent(AbstractAgent):

    def __init__(self,
                 initial_pose: Pose,
                 initial_velocity: Velocity = Velocity(0.0, 0.0)):
        super(RobotAgent, self).__init__(initial_pose, initial_velocity)


@dataclass
class PedestrianForceAgent:
    agent_id: int
    linear_vel_magnitude: float
    radius: float
    mass: float
    inertia: float

    @staticmethod
    def create_default(agent_id: int):
        return PedestrianForceAgent(agent_id=agent_id,
                                    linear_vel_magnitude=1.5,
                                    radius=0.3,
                                    mass=70.0,
                                    inertia=0.5 * 0.3 ** 2)  # 0.5 * r ** 2
