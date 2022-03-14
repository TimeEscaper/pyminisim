from typing import Optional

import numpy as np

from pyminisim.core import AbstractRobotMotionModelState, AbstractRobotMotionModel


class UnicycleRobotModelState(AbstractRobotMotionModelState):
    pass


class UnicycleRobotModel(AbstractRobotMotionModel):

    def __init__(self,
                 initial_pose: np.ndarray,
                 initial_velocity: np.ndarray = np.array([0., 0., 0.]),
                 initial_control: np.ndarray = np.array([0., 0.])):
        assert initial_control.shape == (2,)
        super(UnicycleRobotModel, self).__init__(initial_pose, initial_velocity, initial_control)

    def _init_state(self,
                    initial_pose: np.ndarray,
                    initial_velocity: np.ndarray,
                    initial_control: np.ndarray) -> UnicycleRobotModelState:
        return UnicycleRobotModelState(initial_pose, initial_velocity, initial_control)

    def reset(self,
              initial_pose: np.ndarray,
              initial_velocity: np.ndarray,
              initial_control: np.ndarray):
        assert initial_control.shape == (2,)
        super(UnicycleRobotModel, self).reset(initial_pose, initial_velocity, initial_control)

    def step(self, dt: float, control: Optional[np.ndarray] = None):
        # TODO: Should we update velocity before step?
        if control is None:
            control = self._state.control
        x = self._state.pose[0] + control[0] * np.cos(self._state.pose[2]) * dt
        y = self._state.pose[1] + control[0] * np.sin(self._state.pose[2]) * dt
        theta = self._state.pose[2] + control[1] * dt
        theta = (theta + np.pi) % (2 * np.pi) - np.pi
        v_x = control[0] * np.cos(theta)
        v_y = control[0] * np.sin(theta)
        w = control[1]
        self._state = UnicycleRobotModelState(np.array([x, y, theta]),
                                              np.array([v_x, v_y, w]),
                                              control)
