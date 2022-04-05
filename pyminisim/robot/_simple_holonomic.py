from typing import Optional

import numpy as np

from pyminisim.core import AbstractRobotMotionModelState, AbstractRobotMotionModel


class SimpleHolonomicRobotModelState(AbstractRobotMotionModelState):
    pass


class SimpleHolonomicRobotModel(AbstractRobotMotionModel):
    _CONTROL_DIM = 3

    def __init__(self,
                 initial_pose: np.ndarray,
                 initial_velocity: np.ndarray = np.array([0., 0., 0.]),
                 initial_control: np.ndarray = np.array([0., 0., 0.])):
        assert initial_control.shape == (SimpleHolonomicRobotModel._CONTROL_DIM,)
        super(SimpleHolonomicRobotModel, self).__init__(initial_pose, initial_velocity, initial_control)

    def _init_state(self,
                    initial_pose: np.ndarray,
                    initial_velocity: np.ndarray,
                    initial_control: np.ndarray) -> SimpleHolonomicRobotModelState:
        return SimpleHolonomicRobotModelState(initial_pose, initial_velocity, initial_control)

    def reset(self,
              initial_pose: np.ndarray,
              initial_velocity: np.ndarray,
              initial_control: np.ndarray):
        assert initial_control.shape == (SimpleHolonomicRobotModel._CONTROL_DIM,)
        super(SimpleHolonomicRobotModel, self).reset(initial_pose, initial_velocity, initial_control)

    def step(self, dt: float, control: Optional[np.ndarray] = None):
        if control is None:
            control = self._state.control
        x = self._state.pose[0] + control[0] * dt
        y = self._state.pose[1] + control[1] * dt
        theta = self._state.pose[2] + control[2] * dt
        theta = (theta + np.pi) % (2 * np.pi) - np.pi
        v_x = control[0]
        v_y = control[1]
        w = control[2]
        self._state = SimpleHolonomicRobotModelState(np.array([x, y, theta]),
                                                     np.array([v_x, v_y, w]),
                                                     control)
