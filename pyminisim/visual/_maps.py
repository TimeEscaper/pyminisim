import pygame
import numpy as np

from typing import Tuple

from pyminisim.core import SimulationState
from pyminisim.visual import AbstractMapSkin, VisualizationParams
from pyminisim.visual.util import PoseConverter
from pyminisim.world_map import CirclesWorld, LinesWorld


class EmptyWorldSkin(AbstractMapSkin):

    def __init__(self):
        super(EmptyWorldSkin, self).__init__()

    def render(self, screen, sim_state: SimulationState):
        return


class CirclesWorldSkin(AbstractMapSkin):

    def __init__(self,
                 world_map: CirclesWorld,
                 vis_params: VisualizationParams,
                 color: Tuple[int, int, int] = (0, 255, 0)):
        super(CirclesWorldSkin, self).__init__()
        pose_converter = PoseConverter(vis_params)
        self._pixel_centers = pose_converter.convert(world_map.circles[:, :2])
        self._pixel_radii = [int(radius * vis_params.resolution) for radius in world_map.circles[:, 2]]
        self._color = color

    def render(self, screen, sim_state: SimulationState):
        for center, radius in zip(self._pixel_centers, self._pixel_radii):
            pygame.draw.circle(screen,
                               self._color,
                               (center[0], center[1]),
                               radius,
                               0)


class LinesWorldSkin(AbstractMapSkin):

    def __init__(self,
                 world_map: LinesWorld,
                 vis_params: VisualizationParams,
                 color: Tuple[int, int, int] = (145, 148, 143)):
        super(LinesWorldSkin, self).__init__()
        pose_converter = PoseConverter(vis_params)
        lines = world_map.lines
        n_lines = lines.shape[0]
        self._pixel_lines = np.array(pose_converter.convert(lines.reshape((2 * n_lines, 2)))).reshape((n_lines, 2, 2))
        self._pixel_width = int(vis_params.resolution * world_map.line_width)
        self._color = color

    def render(self, screen, sim_state: SimulationState):
        for i in range(self._pixel_lines.shape[0]):
            line = self._pixel_lines[i]
            pygame.draw.line(screen,
                             self._color,
                             (line[0, 0], line[0, 1]),
                             (line[1, 0], line[1, 1]),
                             self._pixel_width)
