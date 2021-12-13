from abc import ABC
from typing import Tuple
import pkg_resources

import pygame
from pygame.locals import RLEACCEL

from pyminisim.core.util import wrap_angle


class AbstractAgentVisual(pygame.sprite.Sprite, ABC):

    def __init__(self,
                 asset: str,
                 resolution: float,
                 marker_radius_real: float,
                 angle_offset: float):
        super(AbstractAgentVisual, self).__init__()
        self._resolution = resolution
        self._marker_radius = marker_radius_real
        self._angle_offset = angle_offset
        self._surf = pygame.image.load(asset).convert_alpha()
        self._surf.set_colorkey((255, 255, 255), RLEACCEL)
        self._surf = pygame.transform.scale(self._surf, (35, 35))

    def render(self, screen, pixel_pose: Tuple[int, int, int], marker_state: str = "normal"):
        x, y, theta = pixel_pose
        rect = self._surf.get_rect(center=(x, y))
        surf = pygame.transform.rotate(self._surf, int(wrap_angle(self._angle_offset - theta)))
        rect = surf.get_rect(center=rect.center)
        screen.blit(surf, rect)
        if marker_state == "normal":
            color = (225, 227, 230)
        elif marker_state == "detected":
            color = (95, 154, 250)
        elif marker_state == "collision":
            color = (224, 54, 16)
        else:
            raise RuntimeError("Unknown marker state")
        pygame.draw.circle(screen,
                           color,
                           (x, y),
                           int(self._marker_radius * self._resolution),
                           int(0.05 * self._resolution))


class RobotVisual(AbstractAgentVisual):

    _ASSET_PATH = pkg_resources.resource_filename("pyminisim.visual", "assets/robot_3Dred.png")

    def __init__(self, resolution: float):
        super(RobotVisual, self).__init__(RobotVisual._ASSET_PATH, resolution, 0.35, 90.0)


class PedestrianVisual(AbstractAgentVisual):

    _ASSET_PATH = pkg_resources.resource_filename("pyminisim.visual", "assets/character1.png")

    def __init__(self, resolution: float):
        super(PedestrianVisual, self).__init__(PedestrianVisual._ASSET_PATH, resolution, 0.3, 180.0)
