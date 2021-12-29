"""
The Headed Social Force Model (HSFM) implementation.
Based on paper "Walking Ahead: The Headed Social Force Model" by Farina et al.

For increasing computational efficiency, the Numba package is used.
Due to requirements of Numba, some vectorized operations are implemented using standard cycles.
"""



from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
from numba import njit

from pyminisim.core.common import PedestrianForceAgent
from .waypoint import WaypointTracker


@dataclass
class HSFMParams:
    tau: float
    A: float
    B: float
    k_1: float
    k_2: float
    k_o: float
    k_d: float
    k_lambda: float
    alpha: float


DEFAULT_HSFM_PARAMS = HSFMParams(tau=0.5,
                                 A=2000,
                                 B=0.08,
                                 k_1=1.2 * 10 ** 5,
                                 k_2=2.4 * 10 ** 5,
                                 k_o=1.,
                                 k_d=500.,
                                 k_lambda=0.3,
                                 alpha=3.)


@njit
def _hsfm_ode(m: np.ndarray,
              I: np.ndarray,
              v: np.ndarray,
              v_d: np.ndarray,
              r: np.ndarray,
              radii: np.ndarray,
              R: np.ndarray,
              q: np.ndarray,
              robot_radius: float,
              robot_position: np.ndarray,
              robot_linear_vel: np.ndarray,
              tau: float,
              A: float,
              B: float,
              k_1: float,
              k_2: float,
              k_o: float,
              k_d: float,
              k_lambda: float,
              alpha: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    n_pedestrians = m.shape[0]

    # f_0 = m[:, np.newaxis] * (v_d - v) / tau  Omitted due to Numba requirements
    f_0 = np.zeros((n_pedestrians, 2))
    for i in range(f_0.shape[0]):
        f_0[i, :] = m[i] * (v_d[i] - v[i]) / tau

    f_p = np.zeros((n_pedestrians, 2))
    for i in range(n_pedestrians):
        for j in range(n_pedestrians):
            if i == j:
                continue
            f_p[i, :] += _calc_f_p_ij(radii[i], radii[j], r[i], r[j], v[i], v[j],
                                      A, B, k_1, k_2)
        f_p[i, :] += _calc_f_p_ij(radii[i], robot_radius, r[i], robot_position,
                                  v[i], robot_linear_vel, A, B, k_1, k_2)

    f_e = f_p  # Ignore f_w for now

    r_f = R[:, :, 0]
    r_o = R[:, :, 1]
    v_b = _matvec(_inverse(R), v)
    v_o = v_b[:, 1]

    u_f = _dot((f_0 + f_e), r_f)
    u_o = k_o * (_dot(f_e, r_o)) - k_d * v_o
    u_b = np.stack((u_f, u_o), axis=1)

    k_lambda_f_0 = k_lambda * _norm(f_0)
    k_theta = I * k_lambda_f_0
    k_omega = I * (1. + alpha) * np.sqrt(k_lambda_f_0 / alpha)
    theta_0 = np.mod(np.arctan2(v_d[:, 1], v_d[:, 0]), 2 * np.pi)

    u_theta = -k_theta * _wrap_angle(q[:, 0] - theta_0) - k_omega * q[:, 1]

    dr = v  # Equal to the R(theta) * v_b from formulas
    # dv_b = 1. / m[:, np.newaxis] * u_b  Omitted due to Numba requirements
    dv_b = np.zeros((n_pedestrians, 2))
    for i in range(dv_b.shape[0]):
        dv_b[i] = 1. / m[i] * u_b[i]
    dq = np.stack((q[:, 1], u_theta / I), axis=1)  # Equal to the A * dq + b * u_theta from formulas

    # return np.concatenate((dr, dv_b, dq), axis=1).flatten()
    return dr, dv_b, dq


@njit
def _calc_f_p_ij(radius_i: float,
                 radius_j: float,
                 r_i: np.ndarray,
                 r_j: np.ndarray,
                 v_i: np.ndarray,
                 v_j: np.ndarray,
                 A: float,
                 B: float,
                 k_1: float,
                 k_2: float) -> np.ndarray:
    r_ij = radius_i + radius_j
    d_ij = np.linalg.norm(r_i - r_j)
    n_ij = (r_i - r_j) / d_ij
    t_ij = np.array([-n_ij[1], n_ij[0]])
    delta_v_ij = (v_j - v_i) @ t_ij
    g = max(0.0, r_ij - d_ij)

    f_p_ij = (A * np.exp((r_ij - d_ij) / B) + k_1 * g) * n_ij
    f_p_ij += k_2 * g * delta_v_ij * t_ij

    return f_p_ij


@njit
def _inverse(matrices: np.ndarray):
    result = np.zeros_like(matrices)
    for i in range(matrices.shape[0]):
        result[i, :, :] = np.linalg.inv(matrices[i, :, :])
    return result


@njit
def _calc_desired_velocities(current_waypoints: np.ndarray,
                             current_positions: np.ndarray,
                             linear_vel_magnitudes: np.ndarray) -> np.ndarray:
    # Omitted due to Numba requirements
    # direction = current_waypoints - current_positions
    # direction = direction / _norm(direction)[:, np.newaxis]
    # return direction * linear_vel_magnitudes[:, np.newaxis]
    directions = np.zeros_like(current_waypoints)
    for i in range(directions.shape[0]):
        direction = (current_waypoints[i] - current_positions[i])
        direction = direction / np.linalg.norm(direction)
        directions[i, :] = direction * linear_vel_magnitudes[i]
    return directions


@njit
def _matvec(matrix: np.ndarray, vector: np.ndarray) -> np.ndarray:
    # return np.einsum("ijk,ik->ij", matrix, vector)  Omitted due to Numba requirements
    result = np.zeros((matrix.shape[0], matrix.shape[1]))
    for i in range(matrix.shape[0]):
        result[i, :] = matrix[i, :, :] @ vector[i]
    return result


@njit
def _dot(vector1: np.ndarray, vector2: np.ndarray) -> np.ndarray:
    # return np.einsum("ij,ij->i", vector1, vector2)  Omitted due to Numba requirements
    result = np.zeros((vector1.shape[0],))
    for i in range(vector1.shape[0]):
        result[i] = vector1[i] @ vector2[i]
    return result


@njit
def _norm(vectors: np.ndarray):
    result = np.zeros((vectors.shape[0],))
    for i in range(vectors.shape[0]):
        result[i] = np.linalg.norm(vectors[i, :])
    return result


@njit
def _wrap_angle(angle: np.ndarray):
    return (angle + np.pi) % (2 * np.pi) - np.pi


class HeadedSocialForceModel:

    def __init__(self,
                 params: HSFMParams,
                 pedestrians: List[PedestrianForceAgent],
                 initial_poses: np.ndarray,
                 # initial_linear_vels: np.ndarray,
                 # initial_angular_vels: np.ndarray,
                 robot_radius: float,
                 waypoint_tracker: WaypointTracker):
        self._params = params
        self._n_pedestrians = len(pedestrians)
        self._linear_vel_magnitudes = np.array([e.linear_vel_magnitude for e in pedestrians])
        self._waypoint_tracker = waypoint_tracker
        if self._waypoint_tracker.current_waypoints is None:
            self._waypoint_tracker.sample_waypoints(initial_poses[:, :2])

        self._radii = np.array([e.radius for e in pedestrians])
        self._robot_radius = robot_radius
        # Current positions
        self._r = initial_poses[:, :2].copy()
        # Current orientations and angular velocities
        self._q = np.stack((initial_poses[:, 2], np.zeros((self._n_pedestrians,))), axis=1)
        # Current rotation matrix
        self._R = np.array([[[np.cos(theta), -np.sin(theta)],
                             [np.sin(theta), np.cos(theta)]] for theta in initial_poses[:, 2]])
        # Desired velocities v_d
        self._v_d = _calc_desired_velocities(self._waypoint_tracker.current_waypoints, self._r,
                                             self._linear_vel_magnitudes)
        # Current velocities v
        self._v = np.zeros((self._n_pedestrians, 2))
        # Masses and inertia
        self._m = np.array([e.mass for e in pedestrians])
        self._I = np.array([e.inertia for e in pedestrians])

    def update(self, dt: float, robot_position: np.ndarray, robot_linear_vel: np.ndarray) -> np.ndarray:
        self._v_d = _calc_desired_velocities(self._waypoint_tracker.current_waypoints, self._r,
                                             self._linear_vel_magnitudes)

        # Here we do not use "fair" integrators (e.g. scipy.integrate.ode), as it was in original paper's code
        # in sake of better computational performance and Numba compatibility.
        dr, dv_b, dq = _hsfm_ode(self._m, self._I, self._v, self._v_d, self._r,
                                 self._radii, self._R, self._q, self._robot_radius,
                                 robot_position, robot_linear_vel,
                                 **self._params.__dict__)

        self._r = self._r + dr * dt
        self._q = self._q + dq * dt
        self._R = np.array([[[np.cos(theta), -np.sin(theta)],
                             [np.sin(theta), np.cos(theta)]] for theta in self._q[:, 0]])
        self._v = self._v + _matvec(self._R, dv_b * dt)

        self._v_d = _calc_desired_velocities(self._waypoint_tracker.current_waypoints, self._r,
                                             self._linear_vel_magnitudes)

        return np.concatenate([self._r, self._q[:, 0, np.newaxis]], axis=1)
