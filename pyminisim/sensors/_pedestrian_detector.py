from typing import Dict, List, Tuple

import numpy as np

from pyminisim.core import AbstractSensorConfig, AbstractSensorReading, AbstractSensor, WorldState


class PedestrianDetectorConfig(AbstractSensorConfig):

    def __init__(self, max_dist: float, fov: float):
        super(PedestrianDetectorConfig, self).__init__()
        self._max_dist = max_dist
        self._fov = fov

    @property
    def max_dist(self) -> float:
        return self._max_dist

    @property
    def fov(self) -> float:
        return self._fov


class PedestrianDetectorReading(AbstractSensorReading):

    def __init__(self, distances_and_angles: Dict[int, Tuple[float, float]]):
        super(PedestrianDetectorReading, self).__init__()
        self._pedestrians = distances_and_angles

    @property
    def pedestrians(self) -> Dict[int, Tuple[float, float]]:
        return self._pedestrians


class PedestrianDetector(AbstractSensor):

    NAME = "pedestrian_detector"

    def __init__(self,
                 config: PedestrianDetectorConfig = PedestrianDetectorConfig(max_dist=3.,
                                                                             fov=np.deg2rad(30.))):
        super(PedestrianDetector, self).__init__(PedestrianDetector.NAME)
        self._config = config

    @property
    def sensor_config(self) -> AbstractSensorConfig:
        return self._config

    def get_reading(self, world_state: WorldState) -> AbstractSensorReading:
        robot_pose = world_state.robot_pose
        ped_poses = world_state.pedestrians_poses
        readings = {}

        for i, ped in enumerate(ped_poses):
            distance = np.linalg.norm(robot_pose[:2] - ped[:2])
            if distance > self._config.max_dist:
                continue
            point_angle = np.arctan2(ped[1] - robot_pose[1], ped[0] - robot_pose[0])
            angle_diff = (point_angle - robot_pose[2] + np.pi) % (2 * np.pi) - np.pi
            if abs(angle_diff) > self._config.fov / 2.0:
                continue
            readings[i] = (distance, angle_diff)

        return PedestrianDetectorReading(readings)
