from typing import List

import numpy as np

from pyminisim.core.motion import UnicycleMotion
from pyminisim.core.common import RobotAgent, PedestrianAgent


class Simulation:

    PEDESTRIAN_RADIUS = 0.3
    ROBOT_RADIUS = 0.35

    def __init__(self,
                 robot: RobotAgent,
                 pedestrians: List[PedestrianAgent],
                 dt: float):
        robot_pose = robot.pose.to_array()
        robot_vel = robot.velocity.to_array()
        ped_pose = np.array([ped.pose.to_array() for ped in pedestrians])
        ped_vel = np.array([ped.velocity.to_array() for ped in pedestrians])
        self._robot_motion = UnicycleMotion(robot_pose[np.newaxis, :], robot_vel[np.newaxis, :])
        self._ped_motion = UnicycleMotion(ped_pose, ped_vel)
        self._dt = dt

    @property
    def pedestrians_poses(self) -> np.ndarray:
        return self._ped_motion.poses

    @property
    def robot_pose(self) -> np.ndarray:
        return self._robot_motion.poses[0]

    def step(self):
        self._robot_motion.step(self._dt)
        self._ped_motion.step(self._dt)
