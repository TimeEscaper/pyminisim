from abc import ABC, abstractmethod
from typing import Optional, Dict, Tuple

import numpy as np

from ._waypoints import AbstractWaypointTrackerState, AbstractWaypointTracker


class AbstractPedestriansModelState(ABC):

    def __init__(self,
                 pedestrians: Dict[int, Tuple[np.ndarray, np.ndarray]],
                 waypoints_state: Optional[AbstractWaypointTrackerState]):
        # TODO: Move waypoint tracker state to other subclasses
        for ped_id, (pose, velocity) in pedestrians.items():
            assert pose.shape == (3,), f"Each pedestrian pose must be a 3-dim array, but {pose.shape} array was passed for pedestrian {ped_id}"
            assert velocity.shape == (3,), f"Each pedestrian velocity must be a 3-dim array, but {velocity.shape} array was passed for pedestrian {ped_id}"
        self._pedestrians = pedestrians
        self._waypoints = waypoints_state

    @property
    def poses(self) -> Dict[int, np.ndarray]:
        return {k: v[0].copy() for k, v in self._pedestrians.items()}

    @property
    def velocities(self) -> Dict[int, np.ndarray]:
        return {k: v[1].copy() for k, v in self._pedestrians.items()}

    @property
    def waypoints(self) -> Optional[AbstractWaypointTrackerState]:
        return self._waypoints


class AbstractPedestriansModel(ABC):

    @property
    @abstractmethod
    def state(self) -> AbstractPedestriansModelState:
        raise NotImplementedError()

    @abstractmethod
    def step(self, dt: float, robot_pose: Optional[np.ndarray], robot_velocity: Optional[np.ndarray]):
        raise NotImplementedError()

    @abstractmethod
    def reset_to_state(self, state: AbstractPedestriansModelState):
        raise NotImplementedError()
