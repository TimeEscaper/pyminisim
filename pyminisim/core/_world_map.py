from abc import ABC, abstractmethod
from typing import Union

import numpy as np


class AbstractWorldMapState(ABC):
    pass


class AbstractWorldMap(ABC):

    def __init__(self, initial_state: AbstractWorldMapState):
        self._state = initial_state

    @property
    def current_state(self) -> AbstractWorldMapState:
        return self._state

    def reset_to_state(self, state: AbstractWorldMapState):
        self._state = state

    @abstractmethod
    def step(self, dt: float) -> None:
        raise NotImplementedError()

    @abstractmethod
    def closest_distance_to_obstacle(self, point: np.ndarray) -> Union[float, np.ndarray]:
        raise NotImplementedError()

    @abstractmethod
    def is_occupied(self, point: np.ndarray) -> Union[bool, np.ndarray]:
        raise NotImplementedError()

    def is_collision(self, agent_points: np.ndarray, agent_radii: Union[float, np.ndarray],
                     eps_margin: float = 0.05) -> Union[bool, np.ndarray]:
        assert eps_margin >= 0., f"eps_margin must be >= 0., got {eps_margin}"
        if agent_points.shape == (2,):
            single_input = True
            agent_points = agent_points[np.newaxis, :]
            if isinstance(agent_radii, float):
                agent_radii = np.array([agent_radii])
            else:
                assert agent_radii.shape == (1,), \
                    f"agent_radius must be single float or (1,) array for one agent, got {agent_radii.shape}"
        else:
            single_input = False
            assert len(agent_points.shape) == 2 and agent_points.shape[1] == 2, \
                f"agent_point must have shape (n, 2) for n agents, got {agent_points.shape}"
            if isinstance(agent_radii, float):
                agent_radii = np.ones((agent_points.shape[0],)) * agent_radii
            else:
                assert agent_radii.shape == (agent_points.shape[0]), \
                    f"agent_radius shape must match number of agents {agent_points.shape[0]}, got {agent_radii.shape}"

        occupancies = self.is_occupied(agent_points)
        distances = self.closest_distance_to_obstacle(agent_points) - agent_radii + eps_margin
        result = np.logical_or(occupancies, distances <= 0.)
        if single_input:
            result = result[0]

        return result


class AbstractStaticWorldMap(AbstractWorldMap, ABC):

    def __init__(self, initial_state: AbstractWorldMapState):
        super(AbstractStaticWorldMap, self).__init__(initial_state)

    def step(self, dt: float) -> None:
        return
