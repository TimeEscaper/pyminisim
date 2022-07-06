from dataclasses import dataclass
from typing import Dict, Tuple, Optional

import numpy as np

from pyminisim.core import AbstractSensorConfig, AbstractSensorReading, AbstractSensor, WorldState, AbstractWorldMap


@dataclass
class OmniObstacleDetectorNoise:
    distance_mu: float
    distance_sigma: float
    misdetection_prob: float


class OmniObstacleDetectorConfig(AbstractSensorConfig):

    def __init__(self, max_dist: float = np.inf):
        super(OmniObstacleDetectorConfig, self).__init__()
        self._max_dist = max_dist

    @property
    def max_dist(self) -> float:
        return self._max_dist


class OmniObstacleDetectorReading(AbstractSensorReading):

    def __init__(self, min_obstacle_dist: float):
        super(OmniObstacleDetectorReading, self).__init__()
        self._min_obstacle_dist = min_obstacle_dist

    @property
    def min_obstacle_dist(self) -> float:
        return self._min_obstacle_dist

    @property
    def is_finite(self) -> bool:
        return np.isfinite(self._min_obstacle_dist)


class OmniObstacleDetector(AbstractSensor):
    NAME = "omni_obstacle_detector"

    def __init__(self,
                 config: OmniObstacleDetectorConfig = OmniObstacleDetectorConfig(),
                 noise: Optional[OmniObstacleDetectorNoise] = None):
        if noise is not None:
            assert noise.misdetection_prob <= 1.
        super(OmniObstacleDetector, self).__init__(OmniObstacleDetector.NAME, period=0.)
        self._config = config
        self._noise = noise

    @property
    def sensor_config(self) -> AbstractSensorConfig:
        return self._config

    def get_reading(self, world_state: WorldState, world_map: AbstractWorldMap) -> AbstractSensorReading:
        if world_state.robot is None:
            return OmniObstacleDetectorReading(np.inf)
        return OmniObstacleDetectorReading(self._noisify_reading(
            world_map.closest_distance_to_obstacle(world_state.robot.pose[:2])))

    def _noisify_reading(self, distance: float) -> float:
        if self._noise is None:
            return distance
        if not bool(np.random.binomial(1, 1. - self._noise.misdetection_prob)):
            return np.inf
        return distance + np.random.normal(self._noise.distance_mu, self._noise.distance_sigma)
