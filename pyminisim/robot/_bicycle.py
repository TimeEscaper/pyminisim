from typing import Optional

import numpy as np

from pyminisim.core import AbstractRobotMotionModelState, AbstractRobotMotionModel, AbstractWorldMap, ROBOT_RADIUS


class BicycleRobotModelState(AbstractRobotMotionModelState):

    def __init__(self,
                 rear_pose: np.ndarray,
                 control: np.ndarray,
                 wheel_base: float):
        assert rear_pose.shape == (3,)
        assert control.shape == (2,)
        super(BicycleRobotModelState, self).__init__()
        self._rear_pose = rear_pose.copy()
        self._control = control.copy()
        self._wheel_base = wheel_base

    @property
    def pose(self) -> np.ndarray:
        # TODO: Should we return rear pose or center pose?
        return self._rear_pose.copy()

    @property
    def velocity(self) -> np.ndarray:
        # TODO: Should we return rear pose velocity or the center one?
        theta = self._rear_pose[2]
        v = self._control[0]
        delta = self._control[1]
        l = self._wheel_base
        v_x = v * np.cos(theta)
        v_y = v * np.sin(theta)
        w = (v / l) * np.tan(delta)
        return np.array([v_x, v_y, w])

    @property
    def state(self) -> np.ndarray:
        return self._rear_pose.copy()

    @property
    def control(self) -> np.ndarray:
        return self._control.copy()

    @property
    def wheel_base(self) -> float:
        return self._wheel_base


class BicycleRobotModel(AbstractRobotMotionModel):

    def __init__(self,
                 wheel_base: float,
                 initial_center_pose: np.ndarray,
                 initial_control: np.ndarray = np.array([0., 0.])):
        rear_pose = np.array([
            initial_center_pose[0] - (wheel_base / 2.) * np.cos(initial_center_pose[2]),
            initial_center_pose[1] - (wheel_base / 2.) * np.sin(initial_center_pose[2]),
            initial_center_pose[2]
        ])
        state = BicycleRobotModelState(rear_pose, initial_control, wheel_base)
        super(BicycleRobotModel, self).__init__(state)
        self._l = wheel_base

    def step(self, dt: float, world_map: AbstractWorldMap, control: Optional[np.ndarray] = None) -> bool:
        if control is None:
            control = self._state.control

        v = control[0]
        delta = control[1]

        x_rear = self._state.pose[0] + v * np.cos(self._state.pose[2]) * dt
        y_rear = self._state.pose[1] + v * np.sin(self._state.pose[2]) * dt
        theta = self._state.pose[2] + (v / self._l) * np.tan(delta) * dt
        theta = (theta + np.pi) % (2 * np.pi) - np.pi

        d = self._l / 2.
        x_center = x_rear + d * np.cos(theta)
        y_center = y_rear + d * np.sin(theta)

        if world_map.is_collision(np.array([x_center, y_center]), ROBOT_RADIUS):
            return False

        self._state = BicycleRobotModelState(np.array([x_rear, y_rear, theta]),
                                             np.array([v, delta]),
                                             self._l)
        return True
