from dataclasses import dataclass
from typing import Dict, Tuple, Optional

import numpy as np

from pyminisim.core import AbstractSensorConfig, AbstractSensorReading, AbstractSensor, WorldState, AbstractWorldMap
from pyminisim.util import wrap_angle


@dataclass
class PedestrianDetectorNoise:
    distance_mu: float
    distance_sigma: float
    angle_mu: float
    angle_sigma: float
    misdetection_prob: float


class PedestrianDetectorConfig(AbstractSensorConfig):

    RETURN_POLAR = "polar"
    RETURN_ABSOLUTE = "absolute"
    RETURN_RELATIVE = "relative"
    RETURN_RELATIVE_ROTATED = "relative_rotated"

    def __init__(self, max_dist: float, fov: float, return_type: str):
        assert return_type in [PedestrianDetectorConfig.RETURN_POLAR,
                               PedestrianDetectorConfig.RETURN_ABSOLUTE,
                               PedestrianDetectorConfig.RETURN_RELATIVE,
                               PedestrianDetectorConfig.RETURN_RELATIVE_ROTATED]
        super(PedestrianDetectorConfig, self).__init__()
        self._max_dist = max_dist
        self._fov = fov
        self._return_type = return_type

    @property
    def max_dist(self) -> float:
        return self._max_dist

    @property
    def fov(self) -> float:
        return self._fov

    @property
    def return_type(self) -> str:
        return self._return_type


class PedestrianDetectorReading(AbstractSensorReading):

    def __init__(self, poses: Dict[int, Tuple[float, float]]):
        super(PedestrianDetectorReading, self).__init__()
        self._pedestrians = poses

    @property
    def pedestrians(self) -> Dict[int, Tuple[float, float]]:
        return self._pedestrians


class PedestrianDetector(AbstractSensor):
    NAME = "pedestrian_detector"

    def __init__(self,
                 config: PedestrianDetectorConfig = PedestrianDetectorConfig(max_dist=3.,
                                                                             fov=np.deg2rad(30.),
                                                                             return_type=PedestrianDetectorConfig.RETURN_POLAR),
                 noise: Optional[PedestrianDetectorNoise] = None):
        if noise is not None:
            assert noise.misdetection_prob <= 1.
        super(PedestrianDetector, self).__init__(PedestrianDetector.NAME, period=0.)
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

        for i, ped in ped_poses.items():
            distance = np.linalg.norm(robot_pose[:2] - ped[:2])
            if distance > self._config.max_dist:
                continue
            point_angle = np.arctan2(ped[1] - robot_pose[1], ped[0] - robot_pose[0])
            angle_diff = wrap_angle(point_angle - robot_pose[2])
            if abs(angle_diff) > self._config.fov / 2.0:
                continue

            distance, angle_diff = self._noisify_reading(distance, angle_diff)
            reading = self._convert_reading(distance, angle_diff, robot_pose)
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

    def _convert_reading(self, distance: float, angle_diff: float, robot_pose: np.ndarray) -> Tuple[float, float]:
        if self._config.return_type == PedestrianDetectorConfig.RETURN_POLAR:
            return distance, angle_diff

        x_rel_rot = distance * np.cos(angle_diff)
        y_rel_rot = distance * np.sin(angle_diff)
        if self._config.return_type == PedestrianDetectorConfig.RETURN_RELATIVE_ROTATED:
            return x_rel_rot, y_rel_rot

        theta = robot_pose[2]
        x_rel = x_rel_rot * np.cos(theta) - y_rel_rot * np.sin(theta)
        y_rel = x_rel_rot * np.sin(theta) + y_rel_rot * np.cos(theta)
        if self._config.return_type == self._config.RETURN_RELATIVE:
            return x_rel, y_rel

        x_abs = x_rel + robot_pose[0]
        y_abs = y_rel + robot_pose[1]
        if self._config.return_type == self._config.RETURN_ABSOLUTE:
            return x_abs, y_abs

        raise ValueError(f"Unknown return type {self._config.return_type}")
