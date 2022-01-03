from dataclasses import dataclass
from typing import List

import numpy as np


@dataclass
class WorldState:
    robot_pose: np.ndarray
    robot_velocity: np.ndarray
    pedestrians_poses: np.ndarray
    pedestrians_velocities: np.ndarray
    last_control: np.ndarray
    robot_to_pedestrians_collisions: List[int]
