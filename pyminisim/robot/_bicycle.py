from typing import Optional

import numpy as np

from pyminisim.core import AbstractRobotMotionModelState, AbstractRobotMotionModel, AbstractWorldMap, DEFAULT_ROBOT_RADIUS


class BicycleRobotModelState(AbstractRobotMotionModelState):

    def __init__(self,
                 center_pose: np.ndarray,
                 rear_pose: np.ndarray,
                 control: np.ndarray,
                 wheel_base: float):
        assert center_pose.shape == (3,)
        assert rear_pose.shape == (3,)
        assert control.shape == (2,)
        super(BicycleRobotModelState, self).__init__()
        self._center_pose = center_pose.copy()
        self._rear_pose = rear_pose.copy()
        self._control = control.copy()
        self._wheel_base = wheel_base

    @property
    def pose(self) -> np.ndarray:
        # TODO: Should we return rear pose or center pose?
        return self._center_pose.copy()

    @property
    def velocity(self) -> np.ndarray:
        # TODO: Proper velocity calculation
        return np.zeros((3,))

    @property
    def state(self) -> np.ndarray:
        return self._center_pose.copy()

    @property
    def control(self) -> np.ndarray:
        return self._control.copy()

    @property
    def wheel_base(self) -> float:
        return self._wheel_base

    @property
    def rear_pose(self) -> np.ndarray:
        return self._rear_pose.copy()


class BicycleRobotModel(AbstractRobotMotionModel):

    def __init__(self,
                 wheel_base: float,
                 initial_center_pose: np.ndarray,
                 initial_control: np.ndarray = np.array([0., 0.]),
                 robot_radius: float = DEFAULT_ROBOT_RADIUS):
        rear_pose = np.array([
            initial_center_pose[0] - (wheel_base / 2.) * np.cos(initial_center_pose[2]),
            initial_center_pose[1] - (wheel_base / 2.) * np.sin(initial_center_pose[2]),
            initial_center_pose[2]
        ])
        state = BicycleRobotModelState(initial_center_pose, rear_pose, initial_control, wheel_base)
        super(BicycleRobotModel, self).__init__(state, robot_radius)
        self._l = wheel_base

    @property
    def wheel_base(self) -> float:
        return self._l

    def step(self, dt: float, world_map: AbstractWorldMap, control: Optional[np.ndarray] = None) -> bool:
        if control is None:
            control = self._state.control

        v = control[0]
        delta = control[1]

        rear_pose = self._state.rear_pose
        x_rear = rear_pose[0] + v * np.cos(rear_pose[2]) * dt
        y_rear = rear_pose[1] + v * np.sin(rear_pose[2]) * dt
        theta = rear_pose[2] + (v / self._l) * np.tan(delta) * dt
        theta = (theta + np.pi) % (2 * np.pi) - np.pi

        d = self._l / 2.
        x_center = x_rear + d * np.cos(theta)
        y_center = y_rear + d * np.sin(theta)

        if world_map.is_collision(np.array([x_center, y_center]), self._radius):
            return False

        self._state = BicycleRobotModelState(np.array([x_center, y_center, theta]),
                                             np.array([x_rear, y_rear, theta]),
                                             np.array([v, delta]),
                                             self._l)
        return True
