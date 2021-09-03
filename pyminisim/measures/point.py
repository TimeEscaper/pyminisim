import numpy as np


class Point:
    def __init__(self, vector_value):
        assert len(vector_value) == 2
        self._value = np.array([vector_value[0], vector_value[1]])

    @property
    def x(self):
        return self._value[0]

    @property
    def y(self):
        return self._value[1]

    @property
    def vector(self):
        return self._value

    def distance(self, other) -> float:
        return np.linalg.norm(self._value - other.vector)
