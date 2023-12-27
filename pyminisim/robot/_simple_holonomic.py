from typing import Optional

import numpy as np

from pyminisim.core import AbstractRobotMotionModelState, AbstractRobotMotionModel, ROBOT_RADIUS
from pyminisim.core._world_map import AbstractWorldMap


class SimpleHolonomicRobotModelState(AbstractRobotMotionModelState):

    _STATE_DIM = 3
    _CONTROL_DIM = 3

    def __init__(self,
                 state: np.ndarray,
                 control: np.ndarray):
        assert state.shape == (SimpleHolonomicRobotModelState._STATE_DIM,)
        assert control.shape == (SimpleHolonomicRobotModelState._CONTROL_DIM,)
        super(SimpleHolonomicRobotModelState, self).__init__()
        self._state = state.copy()
        self._control = control.copy()

    @property
    def pose(self) -> np.ndarray:
        return self._state.copy()

    @property
    def velocity(self) -> np.ndarray:
        return self._control.copy()

    @property
    def state(self) -> np.ndarray:
        return self._state.copy()

    @property
    def control(self) -> np.ndarray:
        return self._control.copy()


class SimpleHolonomicRobotModel(AbstractRobotMotionModel):

    def __init__(self,
                 initial_pose: np.ndarray,
                 initial_control: np.ndarray = np.array([0., 0., 0.])):
        state = SimpleHolonomicRobotModelState(initial_pose, initial_control)
        super(SimpleHolonomicRobotModel, self).__init__(state)

    def step(self, dt: float, world_map: AbstractWorldMap, control: Optional[np.ndarray] = None) -> bool:
        if control is None:
            control = self._state.control
        x = self._state.pose[0] + control[0] * dt
        y = self._state.pose[1] + control[1] * dt
        theta = self._state.pose[2] + control[2] * dt
        theta = (theta + np.pi) % (2 * np.pi) - np.pi
        v_x = control[0]
        v_y = control[1]
        w = control[2]

        if world_map.is_collision(np.array([x, y]), ROBOT_RADIUS):
            return False

        self._state = SimpleHolonomicRobotModelState(np.array([x, y, theta]),
                                                     np.array([v_x, v_y, w]))

        return True
