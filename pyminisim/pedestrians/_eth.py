from dataclasses import dataclass
from typing import Tuple, Optional, Dict
from pathlib import Path

import numpy as np
from numba import njit

from pyminisim.core import AbstractPedestriansModelState, AbstractPedestriansModel, AbstractWaypointTracker
from pyminisim.core import ROBOT_RADIUS, PEDESTRIAN_RADIUS


class HSFMState(AbstractPedestriansModelState):
    pass


class ETHPedestriansRecord(AbstractPedestriansModel):

    _RECORD_DT = 0.4

    def __init__(self, dt: float, start_frame: int = 0):
        with open("/home/sibirsky/datasets/ETH/ewap_dataset/seq_eth/obsmat.txt") as f:
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
        self._current_t = 0
        self._n_peds = len(ped_ids)
        self._n_steps = current_step
        self._max_time = self._n_steps * ETHPedestriansRecord._RECORD_DT
        self._step_per_gap = int(ETHPedestriansRecord._RECORD_DT // self._dt)


        # n_ids = len(ped_ids)

        # print("ok")

    @property
    def state(self) -> AbstractPedestriansModelState:
        pass

    def step(self, dt: float, robot_pose: Optional[np.ndarray], robot_velocity: Optional[np.ndarray]):
        pass

    def reset_to_state(self, state: AbstractPedestriansModelState):
        pass

    def _get_pedestrians_at_time(self, t: float) -> Dict[int, Tuple[np.ndarray, np.ndarray]]:
        traj_idx = int(t // ETHPedestriansRecord._RECORD_DT)
        offset = int(t // self._step_per_gap) - traj_idx

        traj_item = self._trajectory[traj_idx]
        traj_item_next = self._trajectory[traj_idx + 1]

        result = {}

        for ped_id, ped_state in traj_item.items():
            if ped_id in traj_item_next:
                ped_state_next = traj_item_next[ped_id]
                position = np.linspace(ped_state[:2], ped_state_next[:2], self._step_per_gap)[offset]
            else:
                delta_t = t - traj_idx * ETHPedestriansRecord._RECORD_DT
                position = ped_id[:2] + ped_id[3:5] * delta_t
            pose = np.array([position[0], position[1], 0.])
            velocity = np.array(ped_state[3:])
            result[ped_id] = (pose, velocity)

        return result
