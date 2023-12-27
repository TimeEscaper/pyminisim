from typing import Optional

import numpy as np

from pyminisim.core import AbstractRobotMotionModelState, AbstractRobotMotionModel, AbstractWorldMap, ROBOT_RADIUS


class UnicycleRobotModelState(AbstractRobotMotionModelState):

    _STATE_DIM = 3
    _CONTROL_DIM = 2

    def __init__(self,
                 state: np.ndarray,
                 control: np.ndarray):
        assert state.shape == (UnicycleRobotModelState._STATE_DIM,)
        assert control.shape == (UnicycleRobotModelState._CONTROL_DIM,)
        super(UnicycleRobotModelState, self).__init__()
        self._state = state.copy()
        self._control = control.copy()

    @property
    def pose(self) -> np.ndarray:
        return self._state[:3].copy()

    @property
    def velocity(self) -> np.ndarray:
        theta = self._state[2]
        v = self._control[0]
        v_x = v * np.cos(theta)
        v_y = v * np.sin(theta)
        w = self._control[1]
        return np.array([v_x, v_y, w])

    @property
    def state(self) -> np.ndarray:
        return self._state.copy()

    @property
    def control(self) -> np.ndarray:
        return self._control.copy()


class UnicycleRobotModel(AbstractRobotMotionModel):

    def __init__(self,
                 initial_pose: np.ndarray,
                 initial_control: np.ndarray = np.array([0., 0.])):
        state = UnicycleRobotModelState(initial_pose, initial_control)
        super(UnicycleRobotModel, self).__init__(state)

    def step(self, dt: float, world_map: AbstractWorldMap, control: Optional[np.ndarray] = None) -> bool:
        # TODO: Should we update velocity before step?
        if control is None:
            control = self._state.control
        x = self._state.pose[0] + control[0] * np.cos(self._state.pose[2]) * dt
        y = self._state.pose[1] + control[0] * np.sin(self._state.pose[2]) * dt
        theta = self._state.pose[2] + control[1] * dt
        theta = (theta + np.pi) % (2 * np.pi) - np.pi
        v = control[0]
        w = control[1]

        if world_map.is_collision(np.array([x, y]), ROBOT_RADIUS):
            return False

        self._state = UnicycleRobotModelState(np.array([x, y, theta]),
                                              np.array([v, w]))
        return True
