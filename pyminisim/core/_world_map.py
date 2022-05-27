from abc import ABC, abstractmethod
from typing import Union

import numpy as np


class AbstractWorldMap(ABC):

    @abstractmethod
    def closest_distance_to_obstacle(self, point: np.ndarray) -> Union[float, np.ndarray]:
        raise NotImplementedError()