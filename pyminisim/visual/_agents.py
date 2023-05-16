import math
from abc import ABC
from typing import Tuple
import pkg_resources

import pygame
import numpy as np
from pygame.locals import RLEACCEL

from pyminisim.core import SimulationState, ROBOT_RADIUS, PEDESTRIAN_RADIUS
from pyminisim.visual import VisualizationParams
from pyminisim.visual.util import PoseConverter


class _AbstractAgentSprite(pygame.sprite.Sprite, ABC):

    def __init__(self,
                 asset: str,
                 radius: float,
                 resolution: float):
        super(_AbstractAgentSprite, self).__init__()
        square_side = int(2 * radius * resolution / math.sqrt(2))
        self._surf = pygame.image.load(asset).convert_alpha()
        self._surf = pygame.transform.scale(self._surf, (square_side, square_side))

    def render(self, screen, pixel_pose: Tuple[int, int, int]):
        x, y, theta = pixel_pose
        rect = self._surf.get_rect(center=(x, y))
        surf = pygame.transform.rotate(self._surf, theta)
        rect = surf.get_rect(center=rect.center)
        screen.blit(surf, rect)


class _RobotSprite(_AbstractAgentSprite):

    _ASSET_PACKAGE = "pyminisim.visual"
    _ASSET_PATH = "assets/robot_3Dred.png"

    def __init__(self, resolution: float):
        asset_full_path = pkg_resources.resource_filename(_RobotSprite._ASSET_PACKAGE,
                                                          _RobotSprite._ASSET_PATH)
        super(_RobotSprite, self).__init__(asset_full_path, ROBOT_RADIUS, resolution)


class _PedestrianSprite(_AbstractAgentSprite):

    _ASSET_PACKAGE = "pyminisim.visual"
    _ASSET_PATH = "assets/character1.png"

    def __init__(self, resolution: float):
        asset_full_path = pkg_resources.resource_filename(_PedestrianSprite._ASSET_PACKAGE,
                                                          _PedestrianSprite._ASSET_PATH)
        super(_PedestrianSprite, self).__init__(asset_full_path, PEDESTRIAN_RADIUS, resolution)


class _PedestriansSkin:

    _OFFSET = 180.

    def __init__(self, vis_params: VisualizationParams):
        self._pose_converter = PoseConverter(vis_params)
        self._resolution = vis_params.resolution
        self._sprites = None

    def render(self, screen, sim_state: SimulationState, global_offset: np.ndarray):
        ped_poses = np.stack(list(sim_state.world.pedestrians.poses.values()), axis=0)
        pixel_poses = self._pose_converter.convert(ped_poses,
                                                   global_offset=global_offset,
                                                   angle_offset_degrees=_PedestriansSkin._OFFSET)
        if self._sprites is None or len(pixel_poses) != len(self._sprites):
            self._sprites = [_PedestrianSprite(self._resolution) for _ in range(len(pixel_poses))]
        for pixel_pose, sprite in zip(pixel_poses, self._sprites):
            sprite.render(screen, pixel_pose)


class _RobotSkin:

    _OFFSET = 90.

    def __init__(self, vis_params: VisualizationParams):
        self._pose_converter = PoseConverter(vis_params)
        self._sprite = _RobotSprite(vis_params.resolution)

    def render(self, screen, sim_state: SimulationState, global_offset: np.ndarray):
        self._sprite.render(screen, self._pose_converter.convert(sim_state.world.robot.pose,
                                                                 global_offset=global_offset,
                                                                 angle_offset_degrees=_RobotSkin._OFFSET))
