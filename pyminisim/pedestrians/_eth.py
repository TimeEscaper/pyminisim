import pkg_resources
import numpy as np

from typing import Tuple, Optional, Dict

from pyminisim.core import AbstractPedestriansModelState, AbstractPedestriansModel
from pyminisim.util import wrap_angle, angle_linspace


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

    SEQUENCE_ETH = "eth"
    SEQUENCE_HOTEL = "hotel"

    _RECORD_DT = 0.4

    _CENTER_ETH = (3.86457727, 3.82923192)
    _CENTER_HOTEL = (-0.21777709, -3.30031001)

    def __init__(self, sequence: str, dt: float, start_frame: int = 0):
        if sequence == ETHPedestriansRecord.SEQUENCE_ETH:
            world_center = ETHPedestriansRecord._CENTER_ETH
        elif sequence == ETHPedestriansRecord.SEQUENCE_HOTEL:
            world_center = ETHPedestriansRecord._CENTER_HOTEL
        else:
            raise ValueError(f"Unknown sequence name {sequence}, available options: "
                             f"{ETHPedestriansRecord.SEQUENCE_ETH} and {ETHPedestriansRecord.SEQUENCE_HOTEL}")

        with open(pkg_resources.resource_filename(f"pyminisim.pedestrians", f"assets/obsmat_{sequence}.txt")) as f:
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

            pos_x = -(pos_x - world_center[0])
            pos_y = pos_y - world_center[1]
            v_x = -v_x

            theta = np.arctan2(v_y, v_x)

            # if pedestrian_id not in (5,):
            #     continue

            if frame != current_frame:
                current_step += 1
                current_frame = frame
                trajectory.append({pedestrian_id: np.array([pos_x, pos_y, theta, v_x, v_y, 0.])})
            else:
                trajectory[current_step][pedestrian_id] = np.array([pos_x, pos_y, theta, v_x, v_y, 0.])

            if pedestrian_id not in ped_ids:
                ped_ids.append(pedestrian_id)

        assert start_frame < len(trajectory), \
            f"start_frame must be less than number of frames in sequence which is {len(trajectory)}"

        self._dt = dt
        self._trajectory = trajectory
        self._n_peds = len(ped_ids)
        self._n_frames = current_step
        self._steps_per_gap = int(ETHPedestriansRecord._RECORD_DT // self._dt)

        self._state = ETHState(self._get_pedestrians_at_frame(0, 0), start_frame, 0)

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
                orientation = angle_linspace(ped_state[2], ped_state_next[2], self._steps_per_gap)[sub_step]
                pose = np.array([position[0], position[1], orientation])
                angular_vel = (wrap_angle(ped_state_next[2] - ped_state[2])) / ETHPedestriansRecord._RECORD_DT
            else:
                delta_t = sub_step * self._dt
                position = ped_state[:2] + ped_state[3:5] * delta_t
                orientation = ped_state[2]
                pose = np.array([position[0], position[1], orientation])
                angular_vel = 0.
            velocity = np.array([ped_state[3], ped_state[4], angular_vel])
            result[ped_id] = (pose, velocity)

        return result
