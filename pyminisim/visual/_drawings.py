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


class Covariance2dDrawing(AbstractDrawing):

    NAME = "covariance_2d"

    def __init__(self,
                 mean: np.ndarray,
                 covariance: np.ndarray,
                 color: Tuple[int, int, int],
                 width: float,
                 n_sigma: int = 1):
        super(Covariance2dDrawing, self).__init__()
        assert mean.shape == (2,) and covariance.shape == (2, 2)
        self._mean = mean.copy()
        self._covariance = covariance.copy()
        self._color = color
        self._width = width
        self._n_sigma = n_sigma

    @property
    def name(self) -> str:
        return Covariance2dDrawing.NAME

    @property
    def mean(self) -> np.ndarray:
        return self._mean.copy()

    @property
    def covariance(self) -> np.ndarray:
        return self._covariance.copy()

    @property
    def color(self) -> Tuple[int, int, int]:
        return self._color

    @property
    def width(self) -> float:
        return self._width

    @property
    def n_sigma(self) -> int:
        return self._n_sigma


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


class Covariance2dDrawingRenderer(AbstractDrawingRenderer):

    def __init__(self,
                 drawing: Covariance2dDrawing,
                 vis_params: VisualizationParams):
        # TODO: Drawing type assertions
        pose_converter = PoseConverter(vis_params)

        lambdas, es = np.linalg.eig(drawing.covariance)
        lambdas = np.sqrt(lambdas)

        top_left = np.array(drawing.mean)
        top_left = pose_converter.convert(top_left)
        half_width = int(drawing.n_sigma * lambdas[1] * vis_params.resolution)
        half_height = int(drawing.n_sigma * lambdas[0] * vis_params.resolution)

        rect_width = 2 * half_width
        rect_height = 2 * half_height
        angle = -np.degrees(np.arctan2(*es[:, 0][::-1]))

        rect = (0, 0, rect_width, rect_height)
        thickness = int(vis_params.resolution * drawing.width)

        surface = pygame.Surface((rect_width,
                                  rect_height),
                                 pygame.SRCALPHA, 32)
        surface = surface.convert_alpha()
        ellipse = pygame.draw.ellipse(surface, drawing.color, rect, thickness)
        self._surface = pygame.transform.rotate(surface, angle)

        rect_center = self._surface.get_rect().center
        self._center = (top_left[0] - rect_center[0],
                        top_left[1] - rect_center[1])

    def render(self, screen, sim_state: SimulationState):
        screen.blit(self._surface, self._center)
