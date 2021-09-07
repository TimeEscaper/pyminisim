import numpy as np
from typing import List, Union


class Point:
    def __init__(self, vector_value: Union[List[float], np.ndarray]):
        assert len(vector_value) == 2
        self._value = np.array([vector_value[0], vector_value[1]])

    @property
    def x(self) -> float:
        return self._value[0]

    @property
    def y(self) -> float:
        return self._value[1]

    @property
    def vector(self) -> np.ndarray:
        return self._value

    def distance(self, other) -> float:
        return np.linalg.norm(self._value - other.vector)
