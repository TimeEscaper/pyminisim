from typing import Tuple

import numpy as np
import pygame

from pyminisim.core import SimulationState
from pyminisim.visual import AbstractMapSkin, VisualizationParams
from pyminisim.visual.util import PoseConverter
from pyminisim.world_map import CirclesWorld, EmptyWorld, AABBWorld


class EmptyWorldSkin(AbstractMapSkin):

    def __init__(self):
        super(EmptyWorldSkin, self).__init__()

    def render(self, screen, sim_state: SimulationState, global_offset: np.ndarray):
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
        self._resolution = vis_params.resolution

    def render(self, screen, sim_state: SimulationState, global_offset: np.ndarray):
        pixel_offset_x = -int(self._resolution * global_offset[1])
        pixel_offset_y = int(self._resolution * global_offset[0])
        for center, radius in zip(self._pixel_centers, self._pixel_radii):
            pygame.draw.circle(screen,
                               self._color,
                               (center[0] + pixel_offset_x, center[1] + pixel_offset_y),
                               radius,
                               0)


class AABBWorldSkin(AbstractMapSkin):

    _DEFAULT_COLOR = (0, 255, 0)

    def __init__(self,
                 world_map: AABBWorld,
                 vis_params: VisualizationParams):
        super(AABBWorldSkin, self).__init__()
        pose_converter = PoseConverter(vis_params)
        self._boxes = world_map.boxes

        self._pixel_top_lefts = pose_converter.convert(world_map.boxes[:, :2])
        self._pixel_sizes = [[int(box[0] * vis_params.resolution),
                              int(box[1] * vis_params.resolution)] for box in world_map.boxes[:, 2:]]

        if world_map.colors is not None:
            colors = world_map.colors
            if not isinstance(colors, list):
                colors = [colors for _ in range(world_map.boxes.shape[0])]
        else:
            colors = [AABBWorldSkin._DEFAULT_COLOR for _ in range(world_map.boxes.shape[0])]
        self._colors = colors

        self._resolution = vis_params.resolution

    def render(self, screen, sim_state: SimulationState, global_offset: np.ndarray):
        pixel_offset_x = -int(self._resolution * global_offset[1])
        pixel_offset_y = int(self._resolution * global_offset[0])
        # TODO: Optimize with move_ip
        for top_lefts, sizes, color in zip(self._pixel_top_lefts, self._pixel_sizes, self._colors):
            x = top_lefts[0] + pixel_offset_x
            y = top_lefts[1] + pixel_offset_y
            w = sizes[0]
            h = sizes[1]
            pygame.draw.rect(screen,
                             color,
                             pygame.Rect(x, y, w, h))
