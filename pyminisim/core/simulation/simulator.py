from typing import List

import numpy as np

from pyminisim.core.motion import UnicycleMotion
from pyminisim.core.common import RobotAgent, PedestrianAgent, PedestrianForceAgent
from pyminisim.core.pedestrian import WaypointTracker, DEFAULT_HSFM_PARAMS, HeadedSocialForceModel


class Simulation:

    PEDESTRIAN_RADIUS = 0.3
    ROBOT_RADIUS = 0.35

    def __init__(self,
                 robot: RobotAgent,
                 pedestrians: List[PedestrianAgent],
                 dt: float):
        robot_pose = robot.pose.to_array()
        robot_vel = robot.velocity.to_array()
        self._robot_motion = UnicycleMotion(robot_pose[np.newaxis, :], robot_vel[np.newaxis, :])
        self._dt = dt
        self._pedestrians_poses = np.array([ped.pose.to_array() for ped in pedestrians])
        self._hsfm = HeadedSocialForceModel(DEFAULT_HSFM_PARAMS,
                                            pedestrians=[PedestrianForceAgent.create_default(i)
                                                         for i in range(len(pedestrians))],
                                            initial_poses=self._pedestrians_poses.copy(),
                                            robot_radius=Simulation.ROBOT_RADIUS,
                                            waypoint_tracker=WaypointTracker((7., 7.)))

    @property
    def pedestrians_poses(self) -> np.ndarray:
        return self._pedestrians_poses.copy()

    @property
    def robot_pose(self) -> np.ndarray:
        return self._robot_motion.poses[0]

    def step(self):
        self._pedestrians_poses = self._hsfm.update(self._dt,
                                                    self._robot_motion.poses[0],
                                                    self._robot_motion.velocities[0])
        self._robot_motion.step(self._dt)
