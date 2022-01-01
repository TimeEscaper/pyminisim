from abc import ABC, abstractmethod
from typing import Optional

import numpy as np


class AbstractWaypointTracker(ABC):

    @property
    @abstractmethod
    def current_waypoints(self) -> Optional[np.ndarray]:
        raise NotImplementedError()

    @abstractmethod
    def sample_waypoints(self, agents_poses: np.ndarray) -> np.ndarray:
        raise NotImplementedError()

    @abstractmethod
    def update_waypoints(self, agents_poses: np.ndarray) -> np.ndarray:
        raise NotImplementedError()
