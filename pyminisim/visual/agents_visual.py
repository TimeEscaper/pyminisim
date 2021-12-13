from abc import ABC, abstractmethod
from typing import Tuple
import pkg_resources

import pygame
from pygame.locals import RLEACCEL

from pyminisim.core.common import Pose
from pyminisim.core.util import wrap_angle


class AbstractAgentVisual(pygame.sprite.Sprite, ABC):

    def __init__(self,
                 asset: str,
                 resolution: float,
                 angle_offset: float):
        super(AbstractAgentVisual, self).__init__()
        self._resolution = resolution
        self._angle_offset = angle_offset
        self._surf = pygame.image.load(asset).convert_alpha()
        self._surf.set_colorkey((255, 255, 255), RLEACCEL)
        self._surf = pygame.transform.scale(self._surf, (45, 45))

    # def _pose_transform(self) -> Tuple[int, int, int]:
    #     x = int(self._pose.y * self._resolution)
    #     y = 500 - int(self._pose.x * self._resolution)
    #     theta = int(wrap_angle(-self._pose.theta + self._angle_offset))
    #     return x, y, theta

    def render(self, x: int, y: int, theta: int):
        rect = self._surf.get_rect(center=(x, y))
        surf = pygame.transform.rotate(self._surf, int(wrap_angle(theta + self._angle_offset)))
        rect = surf.get_rect(center=rect.center)
        return surf, rect


class RobotVisual(AbstractAgentVisual):

    _ASSET_PATH = pkg_resources.resource_filename("pyminisim.visual", "assets/robot_3Dred.png")

    def __init__(self, resolution: float):
        super(RobotVisual, self).__init__(RobotVisual._ASSET_PATH, resolution, 90.0)


class PedestrianVisual(AbstractAgentVisual):

    _ASSET_PATH = pkg_resources.resource_filename("pyminisim.visual", "assets/character1.png")

    def __init__(self, resolution: float):
        super(PedestrianVisual, self).__init__(PedestrianVisual._ASSET_PATH, resolution, 180.0)
