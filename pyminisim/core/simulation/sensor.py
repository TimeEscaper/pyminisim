from abc import ABC, abstractmethod
from typing import Any, Dict

import numpy as np

from pyminisim.core.simulation import Simulation
from pyminisim.core.util import wrap_angle


class AbstractSensor(ABC):

    def __init__(self, sensor_name: str):
        self._name = sensor_name

    @property
    def name(self) -> str:
        return self._name

    @abstractmethod
    def get_config(self) -> Dict:
        raise NotImplementedError()

    @abstractmethod
    def get_reading(self, simulation: Simulation) -> Any:
        raise NotImplementedError()


class PedestrianDetector(AbstractSensor):

    NAME = "pedestrian_detector"

    PARAM_MAX_DIST = "max_dist"
    PARAM_FOV = "fov"

    def __init__(self, max_dist: float, fov: float):
        super(PedestrianDetector, self).__init__(PedestrianDetector.NAME)
        self._max_dist = max_dist
        self._fov = fov

    def get_config(self) -> Dict:
        return {PedestrianDetector.PARAM_MAX_DIST: self._max_dist,
                PedestrianDetector.PARAM_FOV: self._fov}

    def get_reading(self, simulation: Simulation) -> Dict:
        robot_pose = simulation.robot_pose
        ped_poses = simulation.pedestrians_poses
        readings = {}

        for i, ped in enumerate(ped_poses):
            distance = np.linalg.norm(robot_pose[:2] - ped[:2])
            if distance > self._max_dist:
                continue
            point_angle = np.rad2deg(np.arctan2(ped[1] - robot_pose[1], ped[0] - robot_pose[0]))
            angle_diff = wrap_angle(point_angle - robot_pose[2])
            if abs(angle_diff) > self._fov / 2.0:
                continue
            readings[i] = (distance, angle_diff)

        return readings
