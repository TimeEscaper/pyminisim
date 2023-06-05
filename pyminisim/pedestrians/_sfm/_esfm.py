import numpy as np

from dataclasses import dataclass
from typing import Optional, Union, Tuple
from numba import jit, njit

from pyminisim.core import AbstractPedestriansModelState, AbstractPedestriansModel, AbstractWaypointTracker
from pyminisim.core import ROBOT_RADIUS, PEDESTRIAN_RADIUS

from ._utils import norm, calc_directions, calc_desired_velocities


# def _calc_bs(current_positions: np.ndarray,
#             current_velocities: np.ndarray,
#             directions: np.ndarray,
#             delta_t: float) -> np.ndarray:
#     r_ab_mat = np.expand_dims(current_positions, 1) - np.expand_dims(current_positions, 0)  # (n_ped, n_ped, 2)
#
#     e_b = np.expand_dims(directions, 0)  # (1, n_ped, 2)
#     v_b = np.expand_dims(norm(current_velocities), 0)  # (1, n_ped)
#
#     r_ab_norm = norm(r_ab_mat)  # (n_ped, n_ped)
#     r_diff_norm = norm(r_ab_mat - v_b * delta_t * e_b)
#     s_b = v_b * delta_t
#
#     b = (r_ab_norm + r_diff_norm) ** 2 - s_b ** 2
#     b = np.fill_diagonal(b, 0.)
#     b = np.sqrt(b) / 2.
#
#     return b


def _calc_goal_forces(current_velocities: np.ndarray,
                      directions: np.ndarray,
                      desired_speeds: np.ndarray,
                      tau: float) -> np.ndarray:
    return (directions * np.expand_dims(desired_speeds, 1) - current_velocities) / tau


def _calc_single_b(ego_position: np.ndarray,
                   target_position: np.ndarray,
                   target_direction: np.ndarray,
                   target_speed: np.ndarray,
                   delta_t: float) -> float:
    r_ab = ego_position - target_position
    diff = r_ab - target_speed * delta_t * target_direction
    b = np.sqrt(np.sum(r_ab ** 2)) + np.sqrt(np.sum(diff ** 2))
    b = 0.5 * np.sqrt(b ** 2 - (target_speed * delta_t) ** 2)
    return b


def _calc_single_v(V_0: float,
                   sigma: float,
                   ego_position: np.ndarray,
                   target_position: np.ndarray,
                   target_direction: np.ndarray,
                   target_speed: np.ndarray,
                   delta_t: float):
    b = _calc_single_b(ego_position,
                       target_position,
                       target_direction,
                       target_speed,
                       delta_t)
    return V_0 * np.exp(-b / sigma)


def _calc_grad_v(V_0: float,
                 sigma: float,
                 ego_position: np.ndarray,
                 target_position: np.ndarray,
                 target_direction: np.ndarray,
                 target_speed: np.ndarray,
                 delta_t: float,
                 delta_diff: float = 1e-3) -> np.ndarray:
    v = _calc_single_v(V_0,
                       sigma,
                       ego_position,
                       target_position,
                       target_direction,
                       target_speed,
                       delta_t)
    dx = np.array([delta_diff, 0.])
    dy = np.array([0., delta_diff])

    dvdx = (_calc_single_v(V_0,
                           sigma,
                           ego_position + dx,
                           target_position,
                           target_direction,
                           target_speed,
                           delta_t) - v) / delta_diff
    dvdy = (_calc_single_v(V_0,
                           sigma,
                           ego_position + dy,
                           target_position,
                           target_direction,
                           target_speed,
                           delta_t) - v) / delta_diff

    return np.array([dvdx, dvdy])


def _calc_repulsive_forces(V_0: float,
                           sigma: float,
                           current_positions: np.ndarray,
                           current_velocities: np.ndarray,
                           directions: np.ndarray,
                           delta_t: float) -> np.ndarray:
    n_peds = current_positions.shape[0]
    forces = np.zeros((n_peds, n_peds, 2), dtype=current_positions.dtype)
    for i in range(n_peds):
        for j in range(n_peds):
            if i == j:
                continue
            target_speed = np.sqrt(current_velocities[j, 0] ** 2 + current_velocities[j, 1] ** 2)
            f_ab = -_calc_grad_v(V_0, sigma, current_positions[i], current_positions[j], directions[j],
                                 target_speed, delta_t)
            forces[i, j, :] = f_ab

    forces = np.sum(forces, axis=1)
    return forces


def _esfm_ode(current_waypoints: np.ndarray,
              current_positions: np.ndarray,
              current_velocities: np.ndarray,
              desired_speeds: np.ndarray,
              tau: float,
              V_0: float,
              sigma: float,
              delta_t: float) -> Tuple[np.ndarray, np.ndarray]:
    directions = calc_directions(current_waypoints, current_positions)
    goal_forces = _calc_goal_forces(current_velocities,
                               directions,
                               desired_speeds,
                               tau)
    repulsive_forces = _calc_repulsive_forces(V_0,
                                              sigma,
                                              current_positions,
                                              current_velocities,
                                              directions,
                                              delta_t)
    forces = goal_forces + repulsive_forces

    new_velocities = current_velocities + forces * delta_t
    new_positions = current_positions + new_velocities * delta_t

    return new_positions, new_velocities


@dataclass
class ESFMParams:
    tau: float = 0.5
    V0: float = 2.1
    sigma: float = 0.3
    v_max: Optional[float] = None


class ESFMState(AbstractPedestriansModelState):
    pass


class ExtendedSocialForceModelPolicy(AbstractPedestriansModel):

    def __init__(self,
                 waypoint_tracker: AbstractWaypointTracker,
                 n_pedestrians: int,
                 initial_poses: Optional[np.ndarray] = None,
                 initial_velocities: Optional[np.ndarray] = None,
                 esfm_params: ESFMParams = ESFMParams(),
                 desired_speeds: Union[np.ndarray, float] = 1.5,
                 robot_visible: bool = True):
        super(ExtendedSocialForceModelPolicy, self).__init__()

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

        self._params = esfm_params
        self._n_pedestrians = initial_poses.shape[0]

        if isinstance(desired_speeds, np.ndarray):
            assert desired_speeds.shape == (n_pedestrians,), \
                "desired_speeds must be single float or (n_pedestrians,) shape ndarray"
            self._desired_speeds = desired_speeds.copy()
        else:
            self._desired_speeds = np.repeat(desired_speeds, self._n_pedestrians)

        self._waypoint_tracker = waypoint_tracker
        if self._waypoint_tracker.state is None:
            self._waypoint_tracker.resample_all({i: initial_poses[i] for i in range(self._n_pedestrians)})
        self._robot_visible = robot_visible

        self._radii = np.repeat(PEDESTRIAN_RADIUS, self._n_pedestrians)
        self._robot_radius = ROBOT_RADIUS

        self._state = ESFMState({i: (initial_poses[i], initial_velocities[i])
                                 for i in range(self._n_pedestrians)}, self._waypoint_tracker.state)

    @property
    def state(self) -> ESFMState:
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

        poses, velocities = _esfm_ode(
            current_waypoints,
            current_poses[:, :2],
            current_vels[:, :2],
            self._desired_speeds,
            self._params.tau,
            self._params.V0,
            self._params.sigma,
            dt
        )

        poses = np.concatenate((poses, current_poses[:, 2, np.newaxis]), axis=-1)
        velocities = np.concatenate((velocities, current_vels[:, 2, np.newaxis]), axis=-1)

        # # Current positions
        # r = current_poses[:, :2] # self._state.poses[:, :2]
        # # Current orientations and angular velocities
        # # q = np.stack((self._state.poses[:, 2], self._state.velocities[:, 2]), axis=1)
        # q = np.stack((current_poses[:, 2], current_vels[:, 2]), axis=1)
        # # Current rotation matrix
        # R = np.array([[[np.cos(theta), -np.sin(theta)],
        #                [np.sin(theta), np.cos(theta)]] for theta in current_poses[:, 2]])
        # # Desired velocities v_d
        # v_d = calc_desired_velocities(current_waypoints, r, self._linear_vel_magnitudes)
        # # Current velocities v
        # # v = self._state.velocities[:, :2]
        # v = current_vels[:, :2]
        #
        # # Here we do not use "fair" integrators (e.g. scipy.integrate.ode), as it was in original paper's code
        # # in sake of better computational performance and Numba compatibility.
        # if robot_pose is not None and self._robot_visible:
        #     robot_pose = robot_pose[:2]
        #     robot_velocity = robot_velocity[:2]
        # else:
        #     robot_pose = None
        #     robot_velocity = None
        # dr, dv_b, dq = _hsfm_ode(self._m, self._I, v, v_d, r, self._radii, R, q, self._robot_radius,
        #                          robot_pose, robot_velocity,
        #                          **self._params.__dict__)
        # if self._noise_std is not None:
        #     dv_b = dv_b + np.random.normal(0, self._noise_std, dv_b.shape)
        # r = r + dr * dt
        # q = q + dq * dt
        # R = np.array([[[np.cos(theta), -np.sin(theta)],
        #                [np.sin(theta), np.cos(theta)]] for theta in q[:, 0]])
        # v = v + _matvec(R, dv_b * dt)
        #
        # poses = np.concatenate([r, q[:, 0, np.newaxis]], axis=1)
        # velocities = np.concatenate([v, q[:, 1, np.newaxis]], axis=1)

        pedestrians = {i: (poses[i, :], velocities[i, :]) for i in range(self._n_pedestrians)}

        waypoints_update = self._waypoint_tracker.update_waypoints({i: poses[i, :] for i in range(self._n_pedestrians)})
        for k, v in waypoints_update.items():
            if v[1]:
                pedestrians[k] = (poses_backup[k].copy(), np.zeros_like(vels_backup[k]))

        self._state = ESFMState(pedestrians, self._waypoint_tracker.state)

    def reset_to_state(self, state: ESFMState):
        self._state = state
        self._waypoint_tracker.reset_to_state(state.waypoints)
