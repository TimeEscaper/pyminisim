from abc import ABC, abstractmethod
from typing import Union, Tuple

import pygame
import numpy as np

from pyminisim.visual import VisualizationParams
from pyminisim.core import SimulationState
from pyminisim.visual.util import PoseConverter


class AbstractDrawing(ABC):

    @property
    @abstractmethod
    def name(self) -> str:
        raise NotImplementedError()


class AbstractDrawingRenderer(ABC):

    @abstractmethod
    def render(self, screen, sim_state: SimulationState):
        raise NotImplementedError()


class CircleDrawing(AbstractDrawing):

    NAME = "circle"

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

    @property
    def name(self) -> str:
        return CircleDrawing.NAME

    @property
    def center(self) -> np.ndarray:
        return self._center.copy()

    @property
    def radius(self) -> float:
        return self._radius

    @property
    def color(self) -> Tuple[int, int, int]:
        return self._color

    @property
    def width(self) -> float:
        return self._width


class CircleDrawingRenderer(AbstractDrawingRenderer):

    def __init__(self,
                 drawing: CircleDrawing,
                 vis_params: VisualizationParams):
        # TODO: Drawing type assertions
        pose_converter = PoseConverter(vis_params)
        self._pixel_center = pose_converter.convert(drawing.center)
        self._pixel_radius = int(drawing.radius * vis_params.resolution)
        self._pixel_width = int(drawing.width * vis_params.resolution)
        self._color = drawing.color

    def render(self, screen, sim_state: SimulationState):
        pygame.draw.circle(screen,
                           self._color,
                           self._pixel_center,
                           self._pixel_radius,
                           self._pixel_width)
