from dataclasses import dataclass
from typing import Dict, Tuple, Optional

import numpy as np

from pyminisim.core import AbstractSensorConfig, AbstractSensorReading, AbstractSensor, WorldState, AbstractWorldMap


# @dataclass
# class OmniObstacleDetectorNoise:
#     distance_mu: float
#     distance_sigma: float
#     misdetection_prob: float


class LidarSensorConfig(AbstractSensorConfig):

    def __init__(self,
                 n_beams: int = 100,
                 max_dist: float = 8.,
                 resolution: float = 0.01,
                 frequency: float = 10.):
        assert n_beams > 0
        assert max_dist > resolution > 0.
        assert frequency > 0.
        super(LidarSensorConfig, self).__init__()
        self._n_beams = n_beams
        self._max_dist = max_dist
        self._resolution = resolution
        self._frequency = frequency

    @property
    def n_beams(self) -> int:
        return self._n_beams

    @property
    def max_dist(self) -> float:
        return self._max_dist

    @property
    def resolution(self) -> float:
        return self._resolution

    @property
    def frequency(self) -> float:
        return self._frequency


class LidarSensorReading(AbstractSensorReading):

    def __init__(self, points: np.ndarray):
        super(LidarSensorReading, self).__init__()
        self._points = points.copy()

    @property
    def points(self) -> np.ndarray:
        return self._points.copy()


class LidarSensor(AbstractSensor):
    NAME = "lidar"

    def __init__(self,
                 config: LidarSensorConfig = LidarSensorConfig()):
        period = 1. / config.frequency if np.isfinite(config.frequency) else 0.
        super(LidarSensor, self).__init__(LidarSensor.NAME, period=period)
        self._config = config

    @property
    def sensor_config(self) -> AbstractSensorConfig:
        return self._config

    def get_reading(self, world_state: WorldState, world_map: AbstractWorldMap) -> AbstractSensorReading:
        if world_state.robot is None:
            return LidarSensorReading(np.array([]))

        center = world_state.robot.pose[:2]

        angles = np.linspace(0., 2 * np.pi, self._config.n_beams)
        distances = np.arange(0., self._config.max_dist, self._config.resolution)
        check_points = np.array([center + np.stack((distances * np.cos(a), distances * np.sin(a)), axis=1)
                                 for a in angles])
        occupation = world_map.is_occupied(check_points.reshape((-1, 2))).reshape((self._config.n_beams, -1))

        points = []
        for i, beam in enumerate(occupation):
            if beam.all() or not beam[0]:
                continue
            points.append(check_points[i, np.argmin(beam), :])

        return LidarSensorReading(np.array(points))
