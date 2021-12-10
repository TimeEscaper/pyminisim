from abc import ABC, abstractmethod
from typing import Tuple

import pygame
from pygame.locals import RLEACCEL

from pyminisim.core.common import Pose
from pyminisim.core.util import wrap_angle


class AbstractAgentVisual(pygame.sprite.Sprite, ABC):

    def __init__(self, initial_pose: Pose, resolution: float, angle_offset: float):
        super(AbstractAgentVisual, self).__init__()
        self._pose = initial_pose
        self._resolution = resolution
        self._angle_offset = angle_offset

    @property
    def pose(self) -> Pose:
        return self._pose

    @pose.setter
    def pose(self, value: Pose):
        self._pose = Pose(value.x, value.y, value.theta)
        self._render()

    def _pose_transform(self) -> Tuple[int, int, int]:
        x = int(self._pose.y * self._resolution)
        y = 500 - int(self._pose.x * self._resolution)
        theta = int(wrap_angle(-self._pose.theta + self._angle_offset))
        return x, y, theta

    @abstractmethod
    def _render(self):
        raise NotImplementedError()


class RobotVisual(AbstractAgentVisual):
    def __init__(self, initial_pose: Pose, resolution: float):
        super(RobotVisual, self).__init__(initial_pose, resolution, 90.0)
        self._surf = pygame.image.load("/home/sibirsky/factory/pyminisim/pyminisim/sprites/robot_3Dred.png").convert_alpha() # pygame.Surface((75, 25))
        self._render()

    @property
    def surf(self):
        return self._surf

    @property
    def rect(self):
        return self._rect

    def _render(self):
        x, y, theta = self._pose_transform()
        self._surf.set_colorkey((255, 255, 255), RLEACCEL)
        self._surf = pygame.transform.scale(self._surf, (45, 45))
        self._rect = self._surf.get_rect(center=(x, y))
        self._surf = pygame.transform.rotate(self._surf, theta)
        self._rect = self._surf.get_rect(center=self._rect.center)


class PedestrianVisual(AbstractAgentVisual):
    def __init__(self, initial_pose: Pose, resolution: float):
        super(PedestrianVisual, self).__init__(initial_pose, resolution, 180.0)
        self._surf = pygame.image.load("/home/sibirsky/factory/pyminisim/pyminisim/sprites/character1.png").convert_alpha() # pygame.Surface((75, 25))
        self._render()

    @property
    def surf(self):
        return self._surf

    @property
    def rect(self):
        return self._rect

    def _render(self):
        x, y, theta = self._pose_transform()
        self._surf.set_colorkey((255, 255, 255), RLEACCEL)
        self._surf = pygame.transform.scale(self._surf, (45, 45))
        self._rect = self._surf.get_rect(center=(x, y))
        self._surf = pygame.transform.rotate(self._surf, theta)
        self._rect = self._surf.get_rect(center=self._rect.center)


# class PedestrianVisual(pygame.sprite.Sprite):
#     def __init__(self):
#         super(PedestrianVisual, self).__init__()
#         self.surf = pygame.image.load("/home/sibirsky/factory/pyminisim/pyminisim/sprites/character1.png").convert_alpha()  # pygame.Surface((75, 25))
#         # self.surf = self.surf.subsurface((541, 200, 100, 100))
#         self.surf.set_colorkey((255, 255, 255), RLEACCEL)
#         self.surf = pygame.transform.scale(self.surf, (45, 45))
#         self.rect = self.surf.get_rect(center=(250, 250))
#
#         self.surf = pygame.transform.rotate(self.surf, 90.0)
#         self.rect = self.surf.get_rect(center=self.rect.center)

