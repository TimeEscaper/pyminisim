import numpy as np

from pyminisim.core import AbstractRobotMotionModel


class UnicycleRobotModel(AbstractRobotMotionModel):

    def __init__(self,
                 initial_pose: np.ndarray,
                 initial_velocity: np.ndarray = np.array([0., 0., 0.]),
                 initial_control: np.ndarray = np.array([0., 0.])):
        assert initial_control.shape == (2,)
        super(UnicycleRobotModel, self).__init__(initial_pose, initial_velocity, initial_control)

    def reset(self,
              initial_pose: np.ndarray,
              initial_velocity: np.ndarray,
              initial_control: np.ndarray):
        assert initial_control.shape == (2,)
        super(UnicycleRobotModel, self).reset(initial_pose, initial_velocity, initial_control)

    def step(self, dt: float):
        # TODO: Should we update velocity before step?
        x = self._pose[0] + self._control[0] * np.cos(self._pose[2]) * dt
        y = self._pose[1] + self._control[0] * np.sin(self._pose[2]) * dt
        theta = self._pose[2] + self._control[1] * dt
        theta = (theta + np.pi) % (2 * np.pi) - np.pi
        v_x = self._control[0] * np.cos(theta)
        v_y = self._control[0] * np.sin(theta)
        w = self._control[1]
        self._pose = np.array([x, y, theta])
        self._velocity = np.array([v_x, v_y, w])
