import numpy as np

from abc import ABC, abstractmethod


class AbstractRobotMotionModelState(ABC):

    @property
    @abstractmethod
    def pose(self) -> np.ndarray:
        """
        Pose of the robot, always in format (x, y, theta).
        :return: NumPy array of [x, y, theta]
        """
        raise NotImplementedError()

    @property
    @abstractmethod
    def velocity(self) -> np.ndarray:
        """
        Linear velocity projections and angular velocity of the robot, always in format (v_x, v_y, w).
        :return: NumPy array of [v_x, v_y, w]
        """
        raise NotImplementedError()

    @property
    @abstractmethod
    def state(self) -> np.ndarray:
        """
        State of the robot according to the motion model.
        :return: NumPy vector of dimensionality according to the model
        """
        raise NotImplementedError()

    @property
    @abstractmethod
    def control(self) -> np.ndarray:
        """
        Control of the robot according to the motion model.
        :return: NumPy vector of dimensionality according to the model
        """
        raise NotImplementedError()
