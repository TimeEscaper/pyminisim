from dataclasses import dataclass
from typing import List, Optional

import numpy as np


@dataclass
class WorldState:
    robot_pose: Optional[np.ndarray]
    robot_velocity: Optional[np.ndarray]
    pedestrians_poses: Optional[np.ndarray]
    pedestrians_velocities: Optional[np.ndarray]
    last_control: Optional[np.ndarray]
    robot_to_pedestrians_collisions: Optional[List[int]]
