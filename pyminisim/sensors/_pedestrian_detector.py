from dataclasses import dataclass
from typing import Dict, Tuple, Optional

import numpy as np

from pyminisim.core import AbstractSensorConfig, AbstractSensorReading, AbstractSensor, WorldState, AbstractWorldMap


@dataclass
class PedestrianDetectorNoise:
    distance_mu: float
    distance_sigma: float
    angle_mu: float
    angle_sigma: float
    misdetection_prob: float


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
                                                                             fov=np.deg2rad(30.)),
                 noise: Optional[PedestrianDetectorNoise] = None):
        if noise is not None:
            assert noise.misdetection_prob <= 1.
        super(PedestrianDetector, self).__init__(PedestrianDetector.NAME)
        self._config = config
        self._noise = noise

    @property
    def sensor_config(self) -> AbstractSensorConfig:
        return self._config

    def get_reading(self, world_state: WorldState, world_map: AbstractWorldMap) -> AbstractSensorReading:
        if world_state.pedestrians is None or world_state.robot is None:
            return PedestrianDetectorReading({})

        robot_pose = world_state.robot.pose
        ped_poses = world_state.pedestrians.poses
        readings = {}

        for i, ped in enumerate(ped_poses):
            distance = np.linalg.norm(robot_pose[:2] - ped[:2])
            if distance > self._config.max_dist:
                continue
            point_angle = np.arctan2(ped[1] - robot_pose[1], ped[0] - robot_pose[0])
            angle_diff = (point_angle - robot_pose[2] + np.pi) % (2 * np.pi) - np.pi
            if abs(angle_diff) > self._config.fov / 2.0:
                continue

            reading = self._noisify_reading(distance, angle_diff)
            if reading is None:
                continue
            readings[i] = reading[0], reading[1]

        return PedestrianDetectorReading(readings)

    def _noisify_reading(self, distance: float, angle_diff: float) -> Optional[Tuple[float, float]]:
        if self._noise is None:
            return distance, angle_diff
        if not bool(np.random.binomial(1, 1. - self._noise.misdetection_prob)):
            return None
        distance = distance + np.random.normal(self._noise.distance_mu, self._noise.distance_sigma)
        angle_diff = angle_diff + np.random.normal(self._noise.angle_mu, self._noise.angle_sigma)
        return distance, angle_diff
