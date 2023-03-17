import numpy as np
from typing import Union


def wrap_angle(angle: Union[np.ndarray, float]) -> float:
    return (angle + np.pi) % (2 * np.pi) - np.pi


def angle_linspace(start_angle: float,
                   end_angle: float,
                   n: int) -> np.ndarray:

    step = wrap_angle(end_angle - start_angle) / n
    result = [start_angle]
    for _ in range(n):
        new_item = result[-1] + step
        if np.abs(new_item) > np.pi:
            if new_item > 0.0:
                new_item = new_item - 2 * np.pi
            else:
                new_item = new_item + 2 * np.pi
        result.append(new_item)
    return np.array(result)
