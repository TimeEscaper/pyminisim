import pygame
from pygame.locals import RLEACCEL


class RobotVisual(pygame.sprite.Sprite):
    def __init__(self):
        super(RobotVisual, self).__init__()
        self.surf = pygame.image.load("/home/sibirsky/Downloads/robopack/PNG/Top view/robot_3Dred.png").convert_alpha() # pygame.Surface((75, 25))
        self.surf.set_colorkey((255, 255, 255), RLEACCEL)
        self.surf = pygame.transform.scale(self.surf, (45, 45))
        self.rect = self.surf.get_rect(center=(250, 250))

        self.surf = pygame.transform.rotate(self.surf, 90.0)
        self.rect = self.surf.get_rect(center=self.rect.center)


class PedestrianVisual(pygame.sprite.Sprite):
    def __init__(self):
        super(PedestrianVisual, self).__init__()
        self.surf = pygame.image.load("/home/sibirsky/Downloads/character1.png").convert_alpha()  # pygame.Surface((75, 25))
        # self.surf = self.surf.subsurface((541, 200, 100, 100))
        self.surf.set_colorkey((255, 255, 255), RLEACCEL)
        self.surf = pygame.transform.scale(self.surf, (45, 45))
        self.rect = self.surf.get_rect(center=(250, 250))

        self.surf = pygame.transform.rotate(self.surf, 90.0)
        self.rect = self.surf.get_rect(center=self.rect.center)

