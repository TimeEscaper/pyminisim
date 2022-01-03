from typing import Tuple

from dataclasses import dataclass


@dataclass
class VisualizationParams:
    screen_size: Tuple[int, int]
    resolution: float
