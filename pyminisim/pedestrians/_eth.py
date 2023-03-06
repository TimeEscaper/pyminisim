from dataclasses import dataclass
from typing import Tuple, Optional, Dict
from pathlib import Path

import numpy as np
from numba import njit

from pyminisim.core import AbstractPedestriansModelState, AbstractPedestriansModel, AbstractWaypointTracker
from pyminisim.core import ROBOT_RADIUS, PEDESTRIAN_RADIUS


class ETHState(AbstractPedestriansModelState):

    def __init__(self,
                 pedestrians: Dict[int, Tuple[np.ndarray, np.ndarray]],
                 current_frame: int,
                 current_sub_step: int):
        super(ETHState, self).__init__(pedestrians=pedestrians, waypoints_state=None)
        self._current_frame = current_frame
        self._current_sub_step = current_sub_step

    @property
    def current_frame(self) -> float:
        return self._current_frame

    @property
    def current_sub_step(self) -> float:
        return self._current_sub_step


class ETHPedestriansRecord(AbstractPedestriansModel):

    _RECORD_DT = 0.4

    _WORLD_BL = np.array([-10.09475671, -10.94118944])
    _WORLD_TR = np.array([19.63625293, 10.36784148])

    def __init__(self, dt: float, start_frame: int = 0):
        with open("/storage/datasets/eth/ewap_dataset/seq_eth/obsmat.txt") as f:
            lines = f.readlines()

        trajectory = []

        current_step = -1
        current_frame = -1
        ped_ids = []
        for line in lines:
            elements = line.split()
            frame = int(float(elements[0]))
            pedestrian_id = int(float(elements[1]))
            pos_x = float(elements[2])
            pos_y = float(elements[4])
            v_x = float(elements[5])
            v_y = float(elements[7])

            pos_x, pos_y = self._transform_pose(pos_x, pos_y)

            if frame != current_frame:
                current_step += 1
                current_frame = frame
                trajectory.append({pedestrian_id: np.array([pos_x, pos_y, 0., v_x, v_y, 0.])})
            else:
                trajectory[current_step][pedestrian_id] = np.array([pos_x, pos_y, 0., v_x, v_y, 0.])

            if pedestrian_id not in ped_ids:
                ped_ids.append(pedestrian_id)

        self._dt = dt
        self._trajectory = trajectory
        self._n_peds = len(ped_ids)
        self._n_frames = current_step
        self._steps_per_gap = int(ETHPedestriansRecord._RECORD_DT // self._dt)

        self._state = ETHState(self._get_pedestrians_at_frame(0, 0), 0, 0)

    @property
    def state(self) -> AbstractPedestriansModelState:
        return self._state

    def step(self, dt: float, robot_pose: Optional[np.ndarray], robot_velocity: Optional[np.ndarray]):
        current_sub_step = self._state.current_sub_step + 1
        if current_sub_step >= self._steps_per_gap:
            current_sub_step = 0
            current_frame = self._state.current_frame + 1
            if current_frame >= self._n_frames:
                current_frame = 0
        else:
            current_frame = self._state.current_frame

        pedestrians = self._get_pedestrians_at_frame(current_frame, current_sub_step)
        self._state = ETHState(pedestrians, current_frame, current_sub_step)

    def reset_to_state(self, state: AbstractPedestriansModelState):
        self._state = state

    def _get_pedestrians_at_frame(self, frame: int, sub_step: int) -> Dict[int, Tuple[np.ndarray, np.ndarray]]:
        traj_item = self._trajectory[frame]
        traj_item_next = self._trajectory[frame + 1]

        result = {}

        for ped_id, ped_state in traj_item.items():
            if ped_id in traj_item_next:
                ped_state_next = traj_item_next[ped_id]
                position = np.linspace(ped_state[:2], ped_state_next[:2], self._steps_per_gap)[sub_step]
            else:
                delta_t = sub_step * self._dt
                position = ped_state[:2] + ped_state[3:5] * delta_t
            pose = np.array([position[0], position[1], 0.])
            velocity = np.array(ped_state[3:])
            result[ped_id] = (pose, velocity)

        return result

    def _transform_pose(self, pose_x: float, pose_y: float) -> Tuple[float, float]:
        pose_x = pose_x - ETHPedestriansRecord._WORLD_BL[0]
        pose_y = pose_y - ETHPedestriansRecord._WORLD_BL[1]

        center = (ETHPedestriansRecord._WORLD_TR - ETHPedestriansRecord._WORLD_BL) / 2
        pose_x = pose_x - center[0]
        pose_y = pose_y - center[1]
        # pose_x = pose_x - ETHPedestriansRecord._WORLD_TL[0]
        # pose_y = pose_y - ETHPedestriansRecord._WORLD_TL[1]
        # pose_y = (ETHPedestriansRecord._WORLD_BR[1] - ETHPedestriansRecord._WORLD_TL[1]) - pose_y
        #
        # center_point = (ETHPedestriansRecord._WORLD_BR - ETHPedestriansRecord._WORLD_TL) / 2
        #
        # pose_x = pose_x - center_point[0]
        # pose_y = pose_y - center_point[1]

        return pose_x, pose_y


