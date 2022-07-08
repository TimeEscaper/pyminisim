from typing import Tuple

import pygame

from pyminisim.core import SimulationState
from pyminisim.visual import AbstractMapSkin, VisualizationParams
from pyminisim.visual.util import PoseConverter
from pyminisim.world_map import CirclesWorld, EmptyWorld


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
