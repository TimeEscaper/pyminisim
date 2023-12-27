from typing import Optional

import numpy as np

from pyminisim.core import AbstractRobotMotionModelState, AbstractRobotMotionModel, ROBOT_RADIUS
from pyminisim.core._world_map import AbstractWorldMap


class DoubleIntegratorRobotModelState(AbstractRobotMotionModelState):
    pass


class DoubleIntegratorRobotModel(AbstractRobotMotionModel):
    _CONTROL_DIM = 2

    def __init__(self,
                 initial_pose: np.ndarray,
                 initial_velocity: Optional[np.ndarray] = None,
                 initial_control: Optional[np.ndarray] = None):
        assert initial_control.shape == (DoubleIntegratorRobotModel._CONTROL_DIM,)
        if initial_velocity is None:
            initial_velocity = np.array([0., 0., 0.])
        if initial_control is None:
            initial_control = np.array([0., 0.])
        super(DoubleIntegratorRobotModel, self).__init__(initial_pose, initial_velocity, initial_control)

    def _init_state(self,
                    initial_pose: np.ndarray,
                    initial_velocity: np.ndarray,
                    initial_control: np.ndarray) -> DoubleIntegratorRobotModelState:
        return DoubleIntegratorRobotModelState(initial_pose, initial_velocity, initial_control)

    def reset(self,
              initial_pose: np.ndarray,
              initial_velocity: np.ndarray,
              initial_control: np.ndarray):
        assert initial_control.shape == (DoubleIntegratorRobotModel._CONTROL_DIM,)
        super(DoubleIntegratorRobotModel, self).reset(initial_pose, initial_velocity, initial_control)

    def step(self, dt: float, world_map: AbstractWorldMap, control: Optional[np.ndarray] = None) -> bool:
        if control is None:
            control = self._state.control
        x = self._state.pose[0] + self._state.velocity[0] * dt
        y = self._state.pose[1] + self._state.velocity[1] * dt
        theta = self._state.pose[2]
        theta = (theta + np.pi) % (2 * np.pi) - np.pi
        v_x = self._state.velocity[0] + control[0] * dt
        v_y = self._state.velocity[1] + control[1] * dt
        w = self._state.velocity[2]

        if world_map.is_collision(np.array([x, y]), ROBOT_RADIUS):
            return False

        self._state = DoubleIntegratorRobotModelState(np.array([x, y, theta]),
                                                      np.array([v_x, v_y, w]),
                                                      control.copy())

        return True
