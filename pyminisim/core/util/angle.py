from typing import Union

import numpy as np


def wrap_angle(angle: Union[float, np.ndarray]):
    return (angle + 180) % 360 - 180
