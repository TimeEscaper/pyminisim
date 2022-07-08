from abc import ABC
from typing import Tuple
import pkg_resources

import pygame
from pygame.locals import RLEACCEL

from pyminisim.core import SimulationState
from pyminisim.visual import VisualizationParams
from pyminisim.visual.util import PoseConverter


class _AbstractAgentSprite(pygame.sprite.Sprite, ABC):

    def __init__(self,
                 asset: str):
                 # marker_radius_real: float,
                 # angle_offset: float):
        super(_AbstractAgentSprite, self).__init__()
        # self._resolution = resolution
        # self._marker_radius = marker_radius_real
        # self._angle_offset = angle_offset
        self._surf = pygame.image.load(asset).convert_alpha()
        self._surf.set_colorkey((255, 255, 255), RLEACCEL)
        self._surf = pygame.transform.scale(self._surf, (35, 35))

    def render(self, screen, pixel_pose: Tuple[int, int, int]):
        x, y, theta = pixel_pose
        rect = self._surf.get_rect(center=(x, y))
        # surf = pygame.transform.rotate(self._surf, int(wrap_angle(self._angle_offset - theta)))
        surf = pygame.transform.rotate(self._surf, theta)
        rect = surf.get_rect(center=rect.center)
        screen.blit(surf, rect)


class _RobotSprite(_AbstractAgentSprite):

    _ASSET_PATH = pkg_resources.resource_filename("pyminisim.visual", "assets/robot_3Dred.png")

    def __init__(self):
        super(_RobotSprite, self).__init__(_RobotSprite._ASSET_PATH)


class _PedestrianSprite(_AbstractAgentSprite):

    _ASSET_PATH = pkg_resources.resource_filename("pyminisim.visual", "assets/character1.png")

    def __init__(self):
        super(_PedestrianSprite, self).__init__(_PedestrianSprite._ASSET_PATH) # 180.0)


class _PedestriansSkin:

    _OFFSET = 180.

    def __init__(self, n_pedestrians: int, vis_params: VisualizationParams):
        self._pose_converter = PoseConverter(vis_params)
        self._sprites = [_PedestrianSprite() for _ in range(n_pedestrians)]

    def render(self, screen, sim_state: SimulationState):
        pixel_poses = self._pose_converter.convert(sim_state.world.pedestrians.poses, _PedestriansSkin._OFFSET)
        for pixel_pose, sprite in zip(pixel_poses, self._sprites):
            sprite.render(screen, pixel_pose)


class _RobotSkin:

    _OFFSET = 90.

    def __init__(self, vis_params: VisualizationParams):
        self._pose_converter = PoseConverter(vis_params)
        self._sprite = _RobotSprite()

    def render(self, screen, sim_state: SimulationState):
        self._sprite.render(screen, self._pose_converter.convert(sim_state.world.robot.pose, _RobotSkin._OFFSET))
