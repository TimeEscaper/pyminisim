from abc import ABC, abstractmethod
from typing import Union, Tuple

import pygame
import numpy as np

from pyminisim.visual import VisualizationParams
from pyminisim.core import SimulationState
from pyminisim.visual.util import convert_pose


class AbstractDrawing(ABC):

    def render(self, screen, sim_state: SimulationState, vis_params: VisualizationParams):
        raise NotImplementedError()


class CircleDrawing(AbstractDrawing):

    def __init__(self,
                 center: Union[Tuple[float, float], np.ndarray],
                 radius: float,
                 color: Tuple[int, int, int],
                 width: float = 0.):
        super(CircleDrawing, self).__init__()
        if not isinstance(center, np.ndarray):
            self._center = np.array(center)
        else:
            self._center = center
        self._radius = radius
        self._color = color
        self._width = width

    def render(self, screen, sim_state: SimulationState, vis_params: VisualizationParams):
        pixel_center = convert_pose(np.array(self._center), vis_params)
        pixel_radius = int(self._radius * vis_params.resolution)
        pixel_width = int(self._width * vis_params.resolution)
        pygame.draw.circle(screen,
                           self._color,
                           pixel_center,
                           pixel_radius,
                           pixel_width)
