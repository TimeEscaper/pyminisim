import numpy as np

from numba import njit


@njit
def norm(vectors: np.ndarray):
    return np.sqrt(np.sum(vectors ** 2, axis=-1))


@njit
def calc_desired_velocities(current_waypoints: np.ndarray,
                            current_positions: np.ndarray,
                            linear_vel_magnitudes: np.ndarray) -> np.ndarray:
    direction = current_waypoints - current_positions
    direction = direction / np.expand_dims(norm(direction), 1)
    return direction * np.expand_dims(linear_vel_magnitudes, 1)
