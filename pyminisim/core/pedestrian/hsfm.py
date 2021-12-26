from dataclasses import dataclass
from typing import List

import numpy as np
from scipy.integrate import ode

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
        self._q = np.concatenate([initial_poses[:, 3], np.zeros((self._n_pedestrians, 2))], axis=1)
        # Current rotation matrix
        self._R = np.array([[[np.cos(theta), -np.sin(theta)],
                             [np.sin(theta), np.cos(theta)]] for theta in initial_poses[:, 3]])
        # Desired velocities v_d
        self._v_d = self._calc_desired_velocities(self._waypoint_tracker.current_waypoints, self._r)
        # Current velocities v
        self._v = np.zeros((self._n_pedestrians, 2))
        # Masses and inertia
        self._m = np.array([e.mass for e in pedestrians])
        self._I = np.array([e.inertia for e in pedestrians])
        # A matrix and b vector
        self._A = np.repeat(np.array([[0., 1.], [0., 0.]]), len(pedestrians), axis=0)
        self._b = np.concatenate([np.zeros((len(pedestrians),)), 1. / self._I], axis=1)

    def update(self, dt: float, robot_position: np.ndarray, robot_linear_vel: np.ndarray) -> np.ndarray:
        solver = ode(lambda t, rp, rv: self._hsfm_ode(t, rp, rv)).set_integrator("dopri5")
        X0 = np.concatenate([self._r, np.linalg.inv(self._R) @ self._v, self._q])
        solver.set_initial_value(X0, 0.)
        solver.set_f_params(robot_position, robot_linear_vel)

        solver.integrate(dt)
        X = solver.y

        self._r = X[:, :2]
        self._q = X[:, 5:]
        self._R = np.array([[[np.cos(theta), -np.sin(theta)],
                             [np.sin(theta), np.cos(theta)]] for theta in self._q[:, 0]])
        self._v = self._R @ X[:, 3:5]

        return np.concatenate([self._r, self._q[:, 0]], axis=1)

    def _hsfm_ode(self, t: float, robot_position: np.ndarray, robot_linear_vel: np.ndarray) -> np.ndarray:
        f_0 = self._m * (self._v_d - self._v) / self._params.tau

        f_p = np.zeros((self._n_pedestrians, 2))
        for i in range(self._n_pedestrians):
            for j in range(self._n_pedestrians):
                if i == j:
                    continue
                f_p[i, :] += self._calc_f_p_ij(self._radii[i], self._radii[j], self._r[i], self._r[j],
                                               self._v[i], self._v[j])
            f_p[i, :] += self._calc_f_p_ij(self._radii[i], self._robot_radius, self._r[i], robot_position,
                                           self._v[i], robot_linear_vel)

        f_e = f_p  # Ignore f_w for now

        r_f = self._R[:, :, 0]
        r_o = self._R[:, :, 1]
        v_b = np.linalg.inv(self._R) @ self._v
        v_o = v_b[:, 1]

        u_f = (f_0 + f_e) @ r_f
        u_o = self._params.k_o * (f_e @ r_o) - self._params.k_d * v_o
        u_b = np.stack([u_f, u_o], axis=1)

        k_lambda_f_0 = self._params.k_lambda * np.linalg.norm(f_0, axis=1)
        k_theta = self._I * k_lambda_f_0
        k_omega = self._I * (1. + self._params.alpha) * np.sqrt(k_lambda_f_0 / self._params.alpha)
        theta_0 = np.mod(np.arctan2(self._v_d[1], self._v_d[0]), 2 * np.pi)

        u_theta = -k_theta * np.unwrap(self._q[:, 0] - theta_0) - k_omega * self._q[:, 1]

        dr = self._v  # Equal to the R(theta) * v_b from formulas
        dv_b = 1. / self._m * u_b
        dq = self._A @ self._q + self._b * u_theta

        return np.concatenate([dr, dv_b, dq], axis=1)

    def _calc_f_p_ij(self,
                     radius_i: float,
                     radius_j: float,
                     r_i: np.ndarray,
                     r_j: np.ndarray,
                     v_i: np.ndarray,
                     v_j: np.ndarray) -> np.ndarray:
        r_ij = radius_i + radius_j
        d_ij = np.linalg.norm(r_i - r_j)
        n_ij = (r_i - r_j) / d_ij
        t_ij = np.array([-n_ij[1], n_ij[0]])
        delta_v_ij = (v_j - v_i) @ t_ij
        g = max(0.0, r_ij - d_ij)

        f_p_ij = (self._params.A * np.exp((r_ij - d_ij) / self._params.B) + self._params.k_1 * g) * n_ij
        f_p_ij += self._params.k_2 * g * delta_v_ij * t_ij

        return f_p_ij

    def _calc_desired_velocities(self,
                                 current_waypoints: np.ndarray,
                                 current_positions: np.ndarray) -> np.ndarray:
        direction = current_waypoints - current_positions
        direction = direction / np.linalg.norm(direction)
        return self._linear_vel_magnitudes * direction
