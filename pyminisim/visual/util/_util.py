from typing import Tuple, Union, List

import numpy as np

from .._visualization_params import VisualizationParams


def convert_pose(simulation_pose: np.ndarray,
                 vis_params: VisualizationParams,
                 angle_offset_degrees: float = 0.) -> Union[Tuple[int, int, int], List[Tuple[int, int, int]]]:
    if len(simulation_pose.shape) == 2:
        return [convert_pose(pose, vis_params, angle_offset_degrees) for pose in simulation_pose]

    sim_x = simulation_pose[0]
    sim_y = simulation_pose[1]
    sim_theta = np.rad2deg(simulation_pose[2])
    x = int(sim_y * vis_params.resolution)
    y = vis_params.screen_size[1] - int(sim_x * vis_params.resolution)
    theta = (angle_offset_degrees - sim_theta + 180.) % 360. - 180.
    return x, y, theta
