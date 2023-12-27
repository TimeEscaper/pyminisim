from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from ._world_map import AbstractWorldMapState
from ._motion import AbstractRobotMotionModelState
from ._pedestrians_model import AbstractPedestriansModelState


@dataclass
class WorldState:
    world_map: AbstractWorldMapState
    robot: Optional[AbstractRobotMotionModelState]
    pedestrians: Optional[AbstractPedestriansModelState]
    robot_to_pedestrians_collisions: Optional[List[int]]
    robot_to_world_collision: bool
