import sys

sys.path.append('..')

import time
from typing import Tuple

import numpy as np
import scipy.optimize
import pygame
import do_mpc
import casadi

from pyminisim.core import Simulation
from pyminisim.world_map import EmptyWorld, CirclesWorld
from pyminisim.robot import UnicycleRobotModel
from pyminisim.pedestrians import HeadedSocialForceModelPolicy, RandomWaypointTracker
from pyminisim.sensors import PedestrianDetectorNoise, PedestrianDetector, OmniObstacleDetector
from pyminisim.visual import Renderer, CircleDrawing


class DoMPCController:

    def __init__(self, dt: float, horizon: int = 30):
        self._dt = dt
        self._horizon = horizon

        self._goal = np.array([0., 0., 0.])

        self._model = do_mpc.model.Model("discrete")

        pose_x = self._model.set_variable(var_type='_x', var_name='pose_x')
        pose_y = self._model.set_variable(var_type='_x', var_name='pose_y')
        pose_theta = self._model.set_variable(var_type='_x', var_name='pose_theta')

        u_v = self._model.set_variable(var_type='_u', var_name='u_v')
        u_omega = self._model.set_variable(var_type='_u', var_name='u_omega')

        self._model.set_rhs('pose_x', pose_x + u_v * casadi.cos(pose_theta) * self._dt)
        self._model.set_rhs('pose_y', pose_y + u_v * casadi.sin(pose_theta) * self._dt)
        self._model.set_rhs('pose_theta', pose_theta + u_omega * self._dt)

        self._model.set_expression(expr_name='cost', expr=casadi.sqrt((pose_x - self._goal[0]) ** 2 +
                                                                      (pose_y - self._goal[1]) ** 2 +
                                                                      (pose_theta - self._goal[2]) ** 2))

        self._model.setup()

        self._mpc = do_mpc.controller.MPC(self._model)
        setup_mpc = {
            'n_robust': 0,
            'n_horizon': 30,
            't_step': 0.1,
            'state_discretization': 'discrete',
            'store_full_solution': True,
            # Use MA27 linear solver in ipopt for faster calculations:
            # 'nlpsol_opts': {'ipopt.linear_solver': 'MA27'}
        }
        self._mpc.set_param(**setup_mpc)

        mterm = self._model.aux['cost']  # terminal cost
        lterm = self._model.aux['cost']  # terminal cost
        # stage costcost

        self._mpc.set_objective(mterm=mterm, lterm=lterm)

        self._mpc.bounds['lower', '_x', 'pose_theta'] = -np.pi
        self._mpc.bounds['upper', '_x', 'pose_theta'] = np.pi
        self._mpc.bounds['lower', '_u', 'u_v'] = -0.5
        self._mpc.bounds['upper', '_u', 'u_v'] = 1.0
        self._mpc.bounds['lower', '_u', 'u_omega'] = -np.deg2rad(25.0)
        self._mpc.bounds['upper', '_u', 'u_omega'] = np.deg2rad(25.0)

        # self._mpc.set_rterm(u=1e-4)  # input penalty

        self._mpc.setup()

        self._estimator = do_mpc.estimator.StateFeedback(self._model)

        self._simulator = do_mpc.simulator.Simulator(self._model)
        self._simulator.set_param(t_step=0.01)
        self._simulator.setup()

    @property
    def dt(self) -> float:
        return self._dt

    @property
    def goal(self) -> np.ndarray:
        return self._goal.copy()

    @goal.setter
    def goal(self, value: np.ndarray):
        self._goal = value.copy()

    def _get_A_matrix(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        return np.eye(3)

    def _get_B_matrix(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        B = np.array([[casadi.cos(x[2]) * self._dt, 0.],
                      [casadi.sin(x[2]) * self._dt, 0.],
                      [0., self._dt]])
        return B

    def _step_cost(self, x: np.ndarray, u: np.ndarray) -> float:
        return np.linalg.norm(x[:2] - self._goal[:2]) + np.linalg.norm(u)

    def predict(self, x_current: np.ndarray) -> np.ndarray:
        self._mpc.x0 = x_current
        self._simulator.x0 = x_current
        self._estimator.x0 = x_current
        # Use initial state to set the initial guess.
        self._mpc.set_initial_guess()

        u0 = self._mpc.make_step(x_current)
        u0 = u0.flatten()

        return u0


class MPCController:

    _STATE_DIM = 3
    _CONTROL_DIM = 2

    def __init__(self, dt: float, horizon: int = 30):
        self._dt = dt
        self._horizon = horizon

        self._lambda_1 = 0.3

        self._goal = np.array([0., 0., 0.])

    @property
    def dt(self) -> float:
        return self._dt

    @property
    def goal(self) -> np.ndarray:
        return self._goal.copy()

    @goal.setter
    def goal(self, value: np.ndarray):
        self._goal = value.copy()

    def predict(self, x_current: np.ndarray) -> np.ndarray:
        return self._optimize(x_current)

    def _forward_dynamics(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        return np.array([x[0] + u[0] * np.cos(x[2]) * self._dt,
                         x[1] + u[0] * np.sin(x[2]) * self._dt,
                         MPCController._wrap_angle(x[2] + u[1] * self._dt)])

    # def _get_dynamics_matrices(self, x: np.ndarray, u: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    #     A = np.eye(3)
    #     B = np.array([[np.cos(x[2]) * self._dt, 0.],
    #                   [np.sin(x[2]) * self._dt, 0.],
    #                   [0., self._dt]])
    #     return A, B

    def _step_cost(self, x: np.ndarray, u: np.ndarray) -> float:
        return self._lambda_1 * np.linalg.norm(x[:2] - self._goal[:2]) + np.linalg.norm(u)

    def _traj_cost(self, x_init: np.ndarray, u_traj: np.ndarray) -> float:
        cost = 0.
        x = x_init
        u_traj = u_traj.reshape((-1, self._CONTROL_DIM))
        for i in range(u_traj.shape[0]):
            u = u_traj[i]
            cost += self._step_cost(x, u)
            x = self._forward_dynamics(x, u)
        return cost

    def _optimize(self, x_init: np.ndarray) -> np.ndarray:
        u_init = np.ones(self._horizon * MPCController._CONTROL_DIM)
        objective = lambda u_traj: self._traj_cost(x_init, u_traj)
        opt_result = scipy.optimize.minimize(objective, u_init, method="Nelder-Mead")
        u_opt = opt_result.x[:MPCController._CONTROL_DIM]
        print(u_opt)
        return u_opt

    @staticmethod
    def _wrap_angle(theta: float) -> float:
        return (theta + np.pi) % (2 * np.pi) - np.pi

def create_sim() -> Tuple[Simulation, Renderer]:
    robot_model = UnicycleRobotModel(initial_pose=np.array([2.0, 3.85, 0.0]),
                                     initial_control=np.array([0.0, np.deg2rad(0.0)]))
    sensor_noise = None
    sensors = [PedestrianDetector(noise=sensor_noise), OmniObstacleDetector()]
    sim = Simulation(sim_dt=0.01,
                     world_map=EmptyWorld(), # CirclesWorld(circles=np.array([[3, 3, 1]])),
                     robot_model=robot_model,
                     pedestrians_model=None,
                     sensors=sensors)
    renderer = Renderer(simulation=sim,
                        resolution=80.0,
                        screen_size=(500, 500))
    return sim, renderer


def main():
    sim, renderer = create_sim()
    renderer.initialize()

    controller = DoMPCController(dt=0.1)
    controller.goal = np.array([1., 1., 0.])
    renderer.draw("goal", CircleDrawing(controller.goal[:2], 0.1, (255, 0, 0), 0))

    running = True
    sim.step()  # First step can take some time due to Numba compilation

    u_pred = np.array([0., 0.])
    hold_time = sim.sim_dt

    while running:
        renderer.render()

        if hold_time >= controller.dt:
            x_current = sim.current_state.world.robot.pose
            u_pred = controller.predict(x_current)
            hold_time = 0.

        sim.step(u_pred)
        hold_time += sim.sim_dt

    # Done! Time to quit.
    pygame.quit()


if __name__ == '__main__':
    main()
