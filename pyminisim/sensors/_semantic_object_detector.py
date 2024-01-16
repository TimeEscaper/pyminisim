import numpy as np

from typing import Optional, Dict
from dataclasses import dataclass
from pyminisim.core import AbstractSensorConfig, AbstractSensorReading, AbstractSensor, WorldState, AbstractWorldMap
from pyminisim.world_map import AABBWorld


class SemanticDetectorConfig(AbstractSensorConfig):

    def __init__(self,
                 max_dist: float = 8.,
                 frequency: float = 10.):
        assert max_dist > 0.
        assert frequency > 0.
        super(SemanticDetectorConfig, self).__init__()
        self._max_dist = max_dist
        self._frequency = frequency

    @property
    def max_dist(self) -> float:
        return self._max_dist

    @property
    def frequency(self) -> float:
        return self._frequency


@dataclass
class SemanticDetection:
    class_name: str
    object_name: Optional[str]
    position: np.ndarray


class SemanticDetectorReading(AbstractSensorReading):

    def __init__(self, detections: Dict[int, SemanticDetection]):
        super(SemanticDetectorReading, self).__init__()
        self._detections = detections

    @property
    def detections(self) -> Dict[int, SemanticDetection]:
        return self._detections


class SemanticDetectorNoise:

    def __init__(self,
                 position_mean: np.ndarray = np.array([0., 0.]),
                 position_cov: np.ndarray = np.diag([0.001, 0.001]),
                 drop_prob: float = 0.):
        assert position_mean.shape == (2,)
        assert position_cov.shape == (2, 2)
        assert (position_cov >= 0.).all()
        assert 0. <= drop_prob <= 1.

        self._position_mean = position_mean
        self._position_cov = position_cov
        self._drop_prob = drop_prob

    @property
    def mean(self) -> np.ndarray:
        return self._position_mean.copy()

    @property
    def cov(self) -> np.ndarray:
        return self._position_cov.copy()

    @property
    def drop_prob(self) -> float:
        return self._drop_prob

    def noisify_reading(self, reading: SemanticDetectorReading) -> SemanticDetectorReading:
        detections = reading.detections
        drop_mask = np.random.binomial(1, 1 - self._drop_prob, len(detections)).astype(bool)
        gaussian_noise = np.random.multivariate_normal(self._position_mean, self._position_cov, len(detections))
        new_detections = {}
        for i, (object_idx, detection) in enumerate(detections.items()):
            if not drop_mask[i]:
                new_detections[object_idx] = SemanticDetection(class_name=detection.class_name,
                                                               object_name=detection.object_name,
                                                               position=detection.position + gaussian_noise[i])
        return SemanticDetectorReading(new_detections)


class SemanticDetector(AbstractSensor):
    NAME = "semantic_detector"

    def __init__(self,
                 config: SemanticDetectorConfig = SemanticDetectorConfig(),
                 noise: Optional[SemanticDetectorNoise] = None):
        period = 1. / config.frequency if np.isfinite(config.frequency) else 0.
        super(SemanticDetector, self).__init__(SemanticDetector.NAME, period=period)
        self._config = config
        self._noise = noise

    @property
    def sensor_config(self) -> AbstractSensorConfig:
        return self._config

    def get_reading(self, world_state: WorldState, world_map: AbstractWorldMap) -> AbstractSensorReading:
        assert isinstance(world_map, AABBWorld), \
            f"Semantic detector can work only with AABBWorld worlds, got instance of {type(world_map)}"

        semantic_objects = world_map.closest_semantic_objects(world_state.robot.pose[:2],
                                                              self._config.max_dist)
        semantic_objects = {k: SemanticDetection(v[0], v[1], v[2]) for k, v in semantic_objects.items()}
        reading = SemanticDetectorReading(semantic_objects)
        if self._noise is not None:
            reading = self._noise.noisify_reading(reading)
        return reading

        # if world_state.robot is None:
        #     return LidarSensorReading(np.array([]))
        #
        # center = world_state.robot.pose[:2]
        #
        # angles = np.linspace(0., 2 * np.pi, self._config.n_beams)
        # distances = np.arange(0., self._config.max_dist, self._config.resolution)
        # check_points = np.array([center + np.stack((distances * np.cos(a), distances * np.sin(a)), axis=1)
        #                          for a in angles])
        # occupation = world_map.is_occupied(check_points.reshape((-1, 2))).reshape((self._config.n_beams, -1))
        #
        # points = []
        # for i, beam in enumerate(occupation):
        #     if beam.any() and not beam[0]:
        #         points.append(check_points[i, np.argmax(beam), :])
        #     # if not beam.all() or not beam[0]:
        #     #     continue
        #     # points.append(check_points[i, np.argmin(beam), :])
        #
        # reading = LidarSensorReading(np.array(points))
        # if self._noise is not None:
        #     reading = self._noise.noisify_reading(reading)
        #
        # return reading
