"""
The Headed Social Force Model (HSFM) implementation.
Based on paper "Walking Ahead: The Headed Social Force Model" by Farina et al.

For increasing computational efficiency, the Numba package is used.
Due to requirements of Numba, some vectorized operations are implemented using standard cycles.
"""


from dataclasses import dataclass
from typing import Tuple, Optional, Union, List

import numpy as np
from numba import njit

from pyminisim.core import AbstractPedestriansModelState, AbstractPedestriansModel, AbstractWaypointTracker
from pyminisim.core import ROBOT_RADIUS, PEDESTRIAN_RADIUS


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

    @staticmethod
    def create_default():
        return HSFMParams(tau=0.2,
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
              robot_position: Optional[np.ndarray],
              robot_linear_vel: Optional[np.ndarray],
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
        if robot_position is not None:
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


class HSFMState(AbstractPedestriansModelState):
    pass


class HeadedSocialForceModelPolicy(AbstractPedestriansModel):

    def __init__(self,
                 waypoint_tracker: AbstractWaypointTracker,
                 n_pedestrians: int,
                 initial_poses: Optional[np.ndarray] = None,
                 initial_velocities: Optional[np.ndarray] = None,
                 hsfm_params: HSFMParams = HSFMParams.create_default(),
                 pedestrian_mass: float = 70.0,
                 pedestrian_linear_velocity_magnitude: Union[float, np.ndarray] = 1.5,
                 robot_visible: bool = True,
                 noise_std: Optional[float] = None):
        super(HeadedSocialForceModelPolicy, self).__init__()

        if initial_poses is None:
            random_positions = waypoint_tracker.sample_independent_points(n_pedestrians, 0.5)
            random_orientations = np.random.uniform(-np.pi, np.pi, size=n_pedestrians)
            initial_poses = np.hstack([random_positions, random_orientations.reshape(-1, 1)])
        else:
            assert initial_poses.shape[0] == n_pedestrians
        if initial_velocities is None:
            initial_velocities = np.zeros((n_pedestrians, 3))
        else:
            assert initial_velocities.shape[0] == n_pedestrians

        self._params = hsfm_params
        self._n_pedestrians = initial_poses.shape[0]
        if isinstance(pedestrian_linear_velocity_magnitude, np.ndarray):
            assert pedestrian_linear_velocity_magnitude.shape == (n_pedestrians,), \
                "Linear velocity magnitude must be float or (n_pedestrians,) shape ndarray"
            self._linear_vel_magnitudes = pedestrian_linear_velocity_magnitude.copy()
        else:
            self._linear_vel_magnitudes = np.repeat(pedestrian_linear_velocity_magnitude, self._n_pedestrians)
        self._waypoint_tracker = waypoint_tracker
        if self._waypoint_tracker.state is None:
            self._waypoint_tracker.resample_all({i: initial_poses[i] for i in range(self._n_pedestrians)})
        self._robot_visible = robot_visible

        self._radii = np.repeat(PEDESTRIAN_RADIUS, self._n_pedestrians)
        self._robot_radius = ROBOT_RADIUS
        # Masses and inertia
        self._m = np.repeat(pedestrian_mass, self._n_pedestrians)
        self._I = 0.5 * (self._radii ** 2)

        self._state = HSFMState({i: (initial_poses[i], initial_velocities[i])
                                 for i in range(self._n_pedestrians)}, self._waypoint_tracker.state)

        self._noise_std = noise_std

    @property
    def state(self) -> HSFMState:
        return self._state

    def step(self, dt: float, robot_pose: Optional[np.ndarray], robot_velocity: Optional[np.ndarray]):
        if robot_pose is None:
            assert robot_velocity is None
        elif robot_velocity is None:
            assert robot_pose is None

        poses_backup = self._state.poses
        vels_backup = self._state.velocities

        current_poses = np.stack(list(self._state.poses.values()), axis=0)
        current_vels = np.stack(list(self._state.velocities.values()), axis=0)
        current_waypoints = np.stack(list(self._waypoint_tracker.state.current_waypoints.values()), axis=0)

        # Current positions
        r = current_poses[:, :2] # self._state.poses[:, :2]
        # Current orientations and angular velocities
        # q = np.stack((self._state.poses[:, 2], self._state.velocities[:, 2]), axis=1)
        q = np.stack((current_poses[:, 2], current_vels[:, 2]), axis=1)
        # Current rotation matrix
        R = np.array([[[np.cos(theta), -np.sin(theta)],
                       [np.sin(theta), np.cos(theta)]] for theta in current_poses[:, 2]])
        # Desired velocities v_d
        v_d = _calc_desired_velocities(current_waypoints, r, self._linear_vel_magnitudes)
        # Current velocities v
        # v = self._state.velocities[:, :2]
        v = current_vels[:, :2]

        # Here we do not use "fair" integrators (e.g. scipy.integrate.ode), as it was in original paper's code
        # in sake of better computational performance and Numba compatibility.
        if robot_pose is not None and self._robot_visible:
            robot_pose = robot_pose[:2]
            robot_velocity = robot_velocity[:2]
        else:
            robot_pose = None
            robot_velocity = None
        dr, dv_b, dq = _hsfm_ode(self._m, self._I, v, v_d, r, self._radii, R, q, self._robot_radius,
                                 robot_pose, robot_velocity,
                                 **self._params.__dict__)
        if self._noise_std is not None:
            dv_b = dv_b + np.random.normal(0, self._noise_std, dv_b.shape)
        r = r + dr * dt
        q = q + dq * dt
        R = np.array([[[np.cos(theta), -np.sin(theta)],
                       [np.sin(theta), np.cos(theta)]] for theta in q[:, 0]])
        v = v + _matvec(R, dv_b * dt)

        poses = np.concatenate([r, q[:, 0, np.newaxis]], axis=1)
        velocities = np.concatenate([v, q[:, 1, np.newaxis]], axis=1)

        pedestrians = {i: (poses[i, :], velocities[i, :]) for i in range(self._n_pedestrians)}

        waypoints_update = self._waypoint_tracker.update_waypoints({i: poses[i, :] for i in range(self._n_pedestrians)})
        for k, v in waypoints_update.items():
            if v[1]:
                pedestrians[k] = (poses_backup[k].copy(), np.zeros_like(vels_backup[k]))

        self._state = HSFMState(pedestrians, self._waypoint_tracker.state)

    def reset_to_state(self, state: HSFMState):
        self._state = state
        self._waypoint_tracker.reset_to_state(state.waypoints)

    def _init_poses(self) -> np.ndarray:
        pass
