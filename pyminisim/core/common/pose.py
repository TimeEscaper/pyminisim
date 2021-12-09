import numpy as np


class Pose:

    def __init__(self, x: float, y: float, theta: float):
        self._x = x
        self._y = y
        self._theta = theta

    @property
    def x(self) -> float:
        return self._x

    @property
    def y(self) -> float:
        return self._y

    @property
    def theta(self) -> float:
        return self._theta

    def to_array(self) -> np.ndarray:
        return np.array(self._x, self._y, self._theta)
