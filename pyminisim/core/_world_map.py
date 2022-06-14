from abc import ABC, abstractmethod
from typing import Union

import numpy as np


class AbstractWorldMap(ABC):

    @abstractmethod
    def closest_distance_to_obstacle(self, point: np.ndarray, radius: float = 0.) -> Union[float, np.ndarray]:
        raise NotImplementedError()
