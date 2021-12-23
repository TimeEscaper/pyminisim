from dataclasses import dataclass
from typing import List

import numpy as np

from pyminisim.core.common import PedestrianForceAgent
from .waypoint import WaypointSampler


@dataclass
class HSFMParams:
    tau: float
    A: float
    B: float
    k_1: float
    k_2: float


class HeadedSocialForceModel:

    def __init__(self,
                 params: HSFMParams,
                 pedestrians: List[PedestrianForceAgent],
                 initial_poses: np.ndarray,
                 initial_linear_vels: np.ndarray,
                 initial_angular_vels: np.ndarray,
                 robot_radius: float,
                 waypoint_sampler: WaypointSampler):
        # assert len(initial_poses.shape) == 2 and initial_poses.shape[1] == 3
        # assert len(desired_velocities) == 2 and desired_velocities.shape[1] == 2 and \
        #        initial_poses.shape[0] == desired_velocities.shape[0]
        self._params = params
        self._n_pedestrians = len(pedestrians)
        self._radii = np.array([e.radius for e in pedestrians])
        self._robot_radius = robot_radius
        # Current positions
        self._r = initial_poses[:, :2].copy()
        # Current orientations and angular velocities
        self._q = np.concatenate([initial_poses[:, 3], initial_angular_vels], axis=1)
        # Desired velocities v_d
        self._v_d = np.array([e.desired_linear_vel for e in pedestrians])
        # Current velocities v
        self._v = initial_linear_vels.copy()
        # Masses and inertia
        self._m = np.array([e.mass for e in pedestrians])
        self._I = np.array([e.inertia for e in pedestrians])
        # A matrix and b vector
        self._A = np.repeat(np.array([[0., 1.], [0., 0.]]), axis=0)
        self._b = np.concatenate([np.zeros((len(pedestrians),)), 1. / self._I], axis=1)

    def update(self, dt: float, robot_pose: np.ndarray):
        pass

    def _hsfm_ode(self, robot_position: np.ndarray, robot_linear_vel: np.ndarray):
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
