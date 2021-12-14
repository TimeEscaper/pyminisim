from abc import ABC
from typing import Tuple

import numpy as np
import pygame
from PIL import Image, ImageDraw

from pyminisim.core.util import wrap_angle


class AbstractSensorVisual(ABC):

    def render(self, screen, pixel_pose: Tuple[int, int, int]):
        raise NotImplementedError()


class PedestrianDetectorVisual(AbstractSensorVisual):

    def __init__(self, resolution: float, max_dist: float, fov: float):
        super(PedestrianDetectorVisual, self).__init__()
        self._resolution = resolution
        self._max_dist = max_dist
        self._fov = fov

        r = int(self._resolution * self._max_dist)
        pil_image = Image.new("RGBA", (2 * r, 2 * r))
        pil_draw = ImageDraw.Draw(pil_image)
        pil_draw.pieslice((0, 0, 2 * r, 2 * r), 0, self._fov, fill=(95, 154, 250, 50))
        mode = pil_image.mode
        size = pil_image.size
        data = pil_image.tobytes()
        self._surf = pygame.image.fromstring(data, size, mode)

    def render(self, screen, pixel_pose: Tuple[int, int, int]):
        x, y, theta = pixel_pose
        theta = int(wrap_angle(90.0 - theta + self._fov / 2.0))
        surf = pygame.transform.rotate(self._surf, theta)
        rect = surf.get_rect(center=(x, y))
        screen.blit(surf, rect)
