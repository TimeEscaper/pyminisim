from typing import Optional

import numpy as np

from pyminisim.core import AbstractSensorConfig, AbstractSensorReading, AbstractSensor, WorldState, AbstractWorldMap


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


class LidarSensorNoise:

    def __init__(self,
                 point_mean: np.ndarray = np.array([0., 0.]),
                 point_cov: np.ndarray = np.diag([0.001, 0.001]),
                 drop_prob: float = 0.):
        assert point_mean.shape == (2,)
        assert point_cov.shape == (2, 2)
        assert (point_cov >= 0.).all()
        assert 0. <= drop_prob <= 1.

        self._point_mean = point_mean
        self._point_cov = point_cov
        self._drop_prob = drop_prob

    @property
    def mean(self) -> np.ndarray:
        return self._point_mean.copy()

    @property
    def cov(self) -> np.ndarray:
        return self._point_cov.copy()

    @property
    def drop_prob(self) -> float:
        return self._drop_prob

    def noisify_reading(self, reading: LidarSensorReading) -> LidarSensorReading:
        points = reading.points
        drop_mask = np.random.binomial(1, 1 - self._drop_prob, points.shape[0]).astype(bool)
        points = points[drop_mask]
        if len(points) == 0:
            return LidarSensorReading(np.array([]))
        gaussian_noise = np.random.multivariate_normal(self._point_mean, self._point_cov, points.shape[0])
        points = points + gaussian_noise
        return LidarSensorReading(points)


class LidarSensor(AbstractSensor):
    NAME = "lidar"

    def __init__(self,
                 config: LidarSensorConfig = LidarSensorConfig(),
                 noise: Optional[LidarSensorNoise] = None):
        period = 1. / config.frequency if np.isfinite(config.frequency) else 0.
        super(LidarSensor, self).__init__(LidarSensor.NAME, period=period)
        self._config = config
        self._noise = noise

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
            if beam.any() and not beam[0]:
                points.append(check_points[i, np.argmax(beam), :])
            # if not beam.all() or not beam[0]:
            #     continue
            # points.append(check_points[i, np.argmin(beam), :])

        reading = LidarSensorReading(np.array(points))
        if self._noise is not None:
            reading = self._noise.noisify_reading(reading)

        return reading
