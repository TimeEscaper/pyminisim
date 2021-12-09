import pygame
from pygame.locals import RLEACCEL

from pyminisim.core.common import Pose


class RobotVisual(pygame.sprite.Sprite):
    def __init__(self, initial_pose: Pose):
        super(RobotVisual, self).__init__()
        self._pose = initial_pose
        self._surf = pygame.image.load("/home/sibirsky/factory/pyminisim/pyminisim/sprites/robot_3Dred.png").convert_alpha() # pygame.Surface((75, 25))
        self._render()

    @property
    def pose(self) -> Pose:
        return self._pose

    @pose.setter
    def pose(self, value: Pose):
        self._pose = Pose(value.x, value.y, value.theta)
        self._render()

    @property
    def surf(self):
        return self._surf

    @property
    def rect(self):
        return self._rect

    def _render(self):
        self._surf.set_colorkey((255, 255, 255), RLEACCEL)
        self._surf = pygame.transform.scale(self._surf, (45, 45))
        self._rect = self._surf.get_rect(center=(int(self._pose.x), int(self._pose.y)))
        self._surf = pygame.transform.rotate(self._surf, int(self._pose.theta))
        self._rect = self._surf.get_rect(center=self._rect.center)


class PedestrianVisual(pygame.sprite.Sprite):
    def __init__(self):
        super(PedestrianVisual, self).__init__()
        self.surf = pygame.image.load("/home/sibirsky/factory/pyminisim/pyminisim/sprites/character1.png").convert_alpha()  # pygame.Surface((75, 25))
        # self.surf = self.surf.subsurface((541, 200, 100, 100))
        self.surf.set_colorkey((255, 255, 255), RLEACCEL)
        self.surf = pygame.transform.scale(self.surf, (45, 45))
        self.rect = self.surf.get_rect(center=(250, 250))

        self.surf = pygame.transform.rotate(self.surf, 90.0)
        self.rect = self.surf.get_rect(center=self.rect.center)

