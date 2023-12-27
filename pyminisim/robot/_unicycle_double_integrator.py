from typing import Optional

import numpy as np

from pyminisim.core import AbstractRobotMotionModelState, AbstractRobotMotionModel, ROBOT_RADIUS
from pyminisim.core._world_map import AbstractWorldMap


class UnicycleDoubleIntegratorRobotModelState(AbstractRobotMotionModelState):

    _STATE_DIM = 5
    _CONTROL_DIM = 2

    def __init__(self,
                 state: np.ndarray,
                 control: np.ndarray) -> None:
        assert state.shape == (UnicycleDoubleIntegratorRobotModelState._STATE_DIM,)
        assert control.shape == (UnicycleDoubleIntegratorRobotModelState._CONTROL_DIM,)
        super(UnicycleDoubleIntegratorRobotModelState, self).__init__()
        self._state = state.copy()
        self._control = control.copy()

    @property
    def pose(self) -> np.ndarray:
        return self._state[:3].copy()

    @property
    def velocity(self) -> np.ndarray:
        theta = self._state[2]
        v = self._state[3]
        v_x = v * np.cos(theta)
        v_y = v * np.sin(theta)
        w = self._state[4]
        return np.array([v_x, v_y, w])

    @property
    def state(self) -> np.ndarray:
        return self._state.copy()

    @property
    def control(self) -> np.ndarray:
        return self._control.copy()


class UnicycleDoubleIntegratorRobotModel(AbstractRobotMotionModel):

    def __init__(self,
                 initial_state: np.ndarray,
                 initial_control: np.ndarray = np.array([0., 0.])) -> None:
        state = UnicycleDoubleIntegratorRobotModelState(initial_state, initial_control)
        super(UnicycleDoubleIntegratorRobotModel, self).__init__(state)

    def step(self, dt: float, world_map: AbstractWorldMap, control: Optional[np.ndarray] = None) -> bool:
        # TODO: Should we update velocity before step?
        if control is None:
            control = self._state.control
        x = self._state.pose[0] + self._state.velocity[0] * dt          # x + vx * dt
        y = self._state.pose[1] + self._state.velocity[1] * dt          # y + vy * dt
        theta = self._state.pose[2] + self._state.velocity[2] * dt      # phi + w * dt
        theta = (theta + np.pi) % (2 * np.pi) - np.pi
        v = self._state.state[3] + control[0] * dt                      # v + a * dt
        w = self._state.state[4] + control[1] * dt                      # w + alpha * dt
        a = control[0]
        alpha = control[1]

        if world_map.is_collision(np.array([x, y]), ROBOT_RADIUS):
            return False

        self._state = UnicycleDoubleIntegratorRobotModelState(np.array([x, y, theta, v, w]),
                                                              np.array([a, alpha]))

        return True
