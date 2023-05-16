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
    def render(self, screen, sim_state: SimulationState, global_offset: np.ndarray):
        raise NotImplementedError()


class CircleDrawing(AbstractDrawing):
    NAME = "circle"

    def __init__(self,
                 center: Union[Tuple[float, float], np.ndarray],
                 radius: float,
                 color: Tuple[int, int, int],
                 width: float = 0.):
        center = np.array(center)
        if len(center.shape) == 1:
            assert center.shape == (2,), f"Center must have shape (2,), {center.shape} is given"
        else:
            assert len(center.shape) == 2 and center.shape[1:] == (2,), f"Center must have shape (n, 2), " \
                                                                        f"{center.shape} is given"
        super(CircleDrawing, self).__init__()
        self._center = center.copy()
        self._radius = radius
        self._color = color
        self._width = width

    @property
    def name(self) -> str:
        return CircleDrawing.NAME

    @property
    def center(self) -> np.ndarray:
        return self._center

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
        if len(mean.shape) == 1:
            assert mean.shape == (2,) and covariance.shape == (2, 2), \
                f"Mean and covariance must have shapes (2,) and (2, 2), {mean.shape} and {covariance.shape} are given"
        else:
            assert len(mean.shape) == 2 and len(covariance.shape) == 3 and mean.shape[0] == covariance.shape[0] \
                   and covariance.shape[1:] == (2, 2), \
                f"Mean and covariance must have shapes (n, 2) and (n, 2, 2), " \
                f"{mean.shape} and {covariance.shape} are given"
        super(Covariance2dDrawing, self).__init__()
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
        return self._mean

    @property
    def covariance(self) -> np.ndarray:
        return self._covariance

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
        center = drawing.center
        if len(center.shape) == 1:
            center = center[np.newaxis, :]
        self._pixel_centers = pose_converter.convert(center)
        self._pixel_radius = int(drawing.radius * vis_params.resolution)
        self._pixel_width = int(drawing.width * vis_params.resolution)
        self._color = drawing.color
        self._resolution = vis_params.resolution

    def render(self, screen, sim_state: SimulationState, global_offset: np.ndarray):
        pixel_offset_x = -int(self._resolution * global_offset[1])
        pixel_offset_y = int(self._resolution * global_offset[0])
        for center_x, center_y in self._pixel_centers:
            pygame.draw.circle(screen,
                               self._color,
                               (center_x + pixel_offset_x, center_y + pixel_offset_y),
                               self._pixel_radius,
                               self._pixel_width)


class Covariance2dDrawingRenderer(AbstractDrawingRenderer):

    def __init__(self,
                 drawing: Covariance2dDrawing,
                 vis_params: VisualizationParams):
        # TODO: Drawing type assertions
        self._resolution = vis_params.resolution
        pose_converter = PoseConverter(vis_params)

        means = drawing.mean
        covariances = drawing.covariance
        if len(means.shape) == 1:
            means = means[np.newaxis, :]
            covariances = covariances[np.newaxis, :, :]

        self._surfaces = []
        self._centers = []

        for mean, covariance in zip(means, covariances):
            lambdas, es = np.linalg.eig(covariance)
            lambdas = np.sqrt(lambdas)

            top_left = np.array(mean)
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
            surface = pygame.transform.rotate(surface, angle)

            rect_center = surface.get_rect().center
            center = (top_left[0] - rect_center[0],
                      top_left[1] - rect_center[1])

            self._surfaces.append(surface)
            self._centers.append(center)

    def render(self, screen, sim_state: SimulationState, global_offset: np.ndarray):
        pixel_offset_x = -int(self._resolution * global_offset[1])
        pixel_offset_y = int(self._resolution * global_offset[0])
        for surface, center in zip(self._surfaces, self._centers):
            screen.blit(surface, (center[0] + pixel_offset_x, center[1] + pixel_offset_y))
