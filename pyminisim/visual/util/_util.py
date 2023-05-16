from typing import Tuple, Union, List, Optional

import numpy as np

from .._visualization_params import VisualizationParams


class PoseConverter:

    def __init__(self,
                 vis_params: VisualizationParams):
        self._resolution = vis_params.resolution
        self._offset_x = vis_params.screen_size[0] // 2
        self._offset_y = vis_params.screen_size[1] // 2

    @property
    def resolution(self) -> float:
        return self._resolution

    def convert(self,
                simulation_pose: np.ndarray,
                angle_offset_degrees: float = 0.,
                global_offset: Optional[Union[Tuple[float, float], np.ndarray]] = None) -> \
            Union[Tuple[int, int, int], List[Tuple[int, int, int]], Tuple[int, int], List[Tuple[int, int]]]:
        if len(simulation_pose.shape) == 2:
            return [self.convert(pose,
                                 angle_offset_degrees=angle_offset_degrees,
                                 global_offset=global_offset) for pose in simulation_pose]

        if global_offset is None:
            global_offset = (0., 0.)
        sim_x = simulation_pose[0] - global_offset[0]
        sim_y = simulation_pose[1] - global_offset[1]
        x = int(sim_y * self._resolution) + self._offset_x
        y = self._offset_y - int(sim_x * self._resolution)
        if simulation_pose.shape[0] == 2:
            return x, y

        sim_theta = np.rad2deg(simulation_pose[2])
        theta = (angle_offset_degrees - sim_theta + 180.) % 360. - 180.
        return x, y, theta
