from typing import Optional, Dict, Tuple
import numpy as np
from pyminisim.core import AbstractPedestriansModelState, AbstractPedestriansModel


class ReplayPedestriansState(AbstractPedestriansModelState):
    
    def __init__(self, step: int, pedestrians: Dict[int, Tuple[np.ndarray, np.ndarray]]):
        super(ReplayPedestriansState, self).__init__(pedestrians, None)
        assert isinstance(step, int) and step >= 0, f"step must be int >= 0, got {step}"
        self._current_step = step

    @property
    def current_step(self) -> int:
        return self._current_step


class ReplayPedestriansPolicy(AbstractPedestriansModel):
    def __init__(self,
                 poses: np.ndarray,
                 velocities: np.ndarray,
                 loop: bool = False) -> None:
        super(ReplayPedestriansPolicy, self).__init__()
        assert len(poses.shape) == 3 and poses.shape[2] == 3, \
            f"Pedestrians poses must have shape (T, n_pedestrians, 3), got {poses.shape}"
        assert len(velocities.shape) == 3 and velocities.shape[2] == 3, \
            f"Pedestrians velocities must have shape (T, n_pedestrians, 3), got {velocities.shape}"
        assert poses.shape[0] == velocities.shape[0] and poses.shape[1] == velocities.shape[1], \
            (f"Poses and velocities must match in time and pedestrians dimension, "
             f"got shapes {poses.shape} and {velocities.shape}")
        self._poses = poses.copy()
        self._velocities = velocities.copy()
        self._loop = loop
        self._max_step = self._poses.shape[0] - 1
        self._n_pedestrians = self._poses.shape[1]
        self._current_step = 0
        self._state = ReplayPedestriansState(self._current_step,
                                             {i: (self._poses[0, i], self._velocities[0, i]) for i in range(self._n_pedestrians)})

    @property
    def state(self) -> ReplayPedestriansState:
        return self._state

    def step(self,
             dt: float = None,  # I left this parameter to save the implementation
             robot_pose: Optional[np.ndarray] = None,
             robot_velocity: Optional[np.ndarray] = None):
        if self._current_step == self._max_step:
            if not self._loop:
                return
            self._current_step = 0
        else:
            self._current_step += 1
        self._state = ReplayPedestriansState(self._current_step,
                                             {i: (self._poses[self._current_step, i],
                                                  self._velocities[self._current_step, i])
                                              for i in range(self._n_pedestrians)})

    def reset_to_state(self, state: ReplayPedestriansState):
        self._state = state
        self._current_step = state.current_step
