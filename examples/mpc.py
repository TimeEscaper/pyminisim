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
from pyminisim.world_map import EmptyWorld, CirclesWorld, AABBWorld, AABBObject
from pyminisim.robot import UnicycleRobotModel
from pyminisim.pedestrians import HeadedSocialForceModelPolicy, RandomWaypointTracker
from pyminisim.sensors import LidarSensor, LidarSensorConfig, SemanticDetector, SemanticDetectorConfig
from pyminisim.visual import Renderer, CircleDrawing


class DoMPCController:

    def __init__(self, dt: float, goal: np.ndarray, sim: Simulation,
                 obstacles: np.ndarray, horizon: int = 20):
        self._dt = dt
        self._sim = sim
        self._horizon = horizon

        self._goal = goal.copy()

        self._model = do_mpc.model.Model("discrete")

        pose_x = self._model.set_variable(var_type='_x', var_name='pose_x')
        pose_y = self._model.set_variable(var_type='_x', var_name='pose_y')
        pose_theta = self._model.set_variable(var_type='_x', var_name='pose_theta')

        u_v = self._model.set_variable(var_type='_u', var_name='u_v')
        u_omega = self._model.set_variable(var_type='_u', var_name='u_omega')

        goal_x = self._model.set_variable('_p', 'goal_x')
        goal_y = self._model.set_variable('_p', 'goal_y')
        goal_theta = self._model.set_variable('_p', 'goal_theta')
        # obstacle_distance = self._model.set_variable(var_type='_tvp', var_name='obstacle_distance')

        self._model.set_rhs('pose_x', pose_x + u_v * casadi.cos(pose_theta) * self._dt)
        self._model.set_rhs('pose_y', pose_y + u_v * casadi.sin(pose_theta) * self._dt)
        self._model.set_rhs('pose_theta', pose_theta + u_omega * self._dt)

        # obstacle_distances = []
        # for i in range(obstacles.shape[0]):
        #     obstacle = obstacles[i]
        #     dist = casadi.sqrt((pose_x - obstacle[0]) ** 2 + (pose_y - obstacle[1]) ** 2) - obstacle[2]
        #     obstacle_distances.append(dist)
        # obstacle_distances = casadi.fmax(0.3 - (casadi.sqrt((pose_x - obstacles[0, 0]) ** 2 + (pose_y - obstacles[0, 1]) ** 2) - obstacles[0, 2] - 0.35), 0.)
        obstacle_distance = (casadi.sqrt((pose_x - obstacles[0, 0]) ** 2 + (pose_y - obstacles[0, 1]) ** 2) - obstacles[0, 2] - 0.35)
        self._model.set_expression("obstacle_distance", obstacle_distance)

        # print((np.sqrt((sim.current_state.world.robot.pose[0] - obstacles[0, 0]) ** 2 + (sim.current_state.world.robot.pose[1] - obstacles[0, 1]) ** 2)) - obstacles[0, 2] - 0.35)
        # print(np.linalg.norm(sim.current_state.world.robot.pose[:2] - obstacles[0, :2]))

        self._model.set_expression(expr_name='cost',
                                   expr=casadi.sqrt((pose_x - goal[0]) ** 2 +
                                                    (pose_y - goal[1]) ** 2 +
                                                    (pose_theta - goal[2]) ** 2) ** 2)
                                        # casadi.mmin([(pose_x)]))
                                        # obstacle_distance ** 3)

        self._model.setup()

        self._mpc = do_mpc.controller.MPC(self._model)
        setup_mpc = {
            'n_robust': 0,
            'n_horizon': horizon,
            't_step': 0.1,
            'state_discretization': 'discrete',
            'store_full_solution': True,
            "nlpsol_opts": {"ipopt.print_level": 0,
                            "ipopt.sb": "yes",
                            "print_time": 0}
            # Use MA27 linear solver in ipopt for faster calculations:
            # 'nlpsol_opts': {'ipopt.linear_solver': 'MA27'}
        }
        self._mpc.set_param(**setup_mpc)

        # controller_tvp_template = self._mpc.get_tvp_template()
        # self._mpc.set_tvp_fun(lambda t_now, template=controller_tvp_template:
        #                       self._tvp_fun_controller(template, t_now))

        mterm = self._model.aux['cost']  # terminal cost
        lterm = self._model.aux['cost']  # terminal cost
        # stage costcost

        self._mpc.set_objective(mterm=mterm, lterm=lterm)

        self._mpc.bounds['lower', '_x', 'pose_theta'] = -np.pi
        self._mpc.bounds['upper', '_x', 'pose_theta'] = np.pi
        self._mpc.bounds['lower', '_u', 'u_v'] = 0.
        self._mpc.bounds['upper', '_u', 'u_v'] = 1.8
        self._mpc.bounds['lower', '_u', 'u_omega'] = -np.deg2rad(50.)
        self._mpc.bounds['upper', '_u', 'u_omega'] = np.deg2rad(50.)

        # self._mpc.set_rterm(u=1e-4)  # input penalty

        self._mpc.set_uncertainty_values(goal_x=np.array([goal[0]]),
                                         goal_y=np.array([goal[1]]),
                                         goal_theta=np.array([goal[2]]))

        # self._mpc.set_nl_cons("obstacle_distances", obstacle_distances, 0)
        self._mpc.set_nl_cons('obstacles', -self._model.aux['obstacle_distance'], 0)

        self._mpc.setup()

        self._mpc_estimator = do_mpc.estimator.StateFeedback(self._model)

        self._mpc_simulator = do_mpc.simulator.Simulator(self._model)
        self._mpc_simulator.set_param(t_step=0.01)
        p_num_simulator = self._mpc_simulator.get_p_template()
        self._mpc_simulator.set_p_fun(lambda t_now, goal=goal, p_template=p_num_simulator: self._p_fun_simulator(goal, p_template, t_now))
        # simulator_tvp_template = self._mpc_simulator.get_tvp_template()
        # self._mpc_simulator.set_tvp_fun(lambda t_now, template=simulator_tvp_template: self._tvp_fun_simulator(template, t_now))
        self._mpc_simulator.setup()

    @property
    def dt(self) -> float:
        return self._dt

    @property
    def goal(self) -> np.ndarray:
        return self._goal.copy()

    def _p_fun_simulator(self, goal, p_template, t_now):
        p_template["goal_x"] = goal[0]
        p_template["goal_y"] = goal[1]
        p_template["goal_theta"] = goal[2]
        return p_template

    def _tvp_fun_controller(self, tvp_template, t_now):
        x = self._mpc.x0
        dist = 0.3 - casadi.sqrt((x[0] - 3.) ** 2 + (x[1] - 3.) ** 2) - 0.5
        if dist < 0.:
            dist = 0.
        if t_now != 0:
            print(t_now)
        for i in range(self._horizon + 1):
            tvp_template["_tvp", i, "obstacle_distance"] = dist
        return tvp_template

    def _tvp_fun_simulator(self, tvp_template, t_now):
        tvp_template["obstacle_distance"] = 0.
        return tvp_template

    def predict(self, x_current: np.ndarray) -> np.ndarray:
        self._mpc.set_uncertainty_values(goal_x=np.array([self._goal[0]]),
                                         goal_y=np.array([self._goal[1]]),
                                         goal_theta=np.array([self._goal[2]]))
        self._mpc.x0 = x_current
        self._mpc_simulator.x0 = x_current
        self._mpc_estimator.x0 = x_current
        # Use initial state to set the initial guess.
        self._mpc.set_initial_guess()

        u0 = self._mpc.make_step(x_current)
        u0 = u0.flatten()

        return u0


OBSTACLES = np.array([[1.5, 0., 0.8]])


def create_walls() -> AABBWorld:
    thickness = 0.3
    walls = [
        (2.5, -2.5, 5., thickness),
        (2.5 - thickness, -2.5, thickness, 5. - thickness),
        (2.5 - thickness, 2.5 - thickness, thickness, 5. - thickness),
        (-2.5 + thickness, -2.5, 5., thickness)
    ]
    table = (1.5, 1., 0.7, 0.7)
    objects = [AABBObject(e, "wall", f"wall_{i}", (199, 195, 195)) for i, e in enumerate(walls)]
    objects.append(AABBObject(table, "table", "table", (131, 235, 52)))
    return AABBWorld(objects=objects)


def create_sim() -> Tuple[Simulation, Renderer]:
    robot_model = UnicycleRobotModel(initial_pose=np.array([0., 0., 0.]),
                                     initial_control=np.array([0., np.deg2rad(0.)]))
    sensors = [
        LidarSensor(config=LidarSensorConfig()),
        SemanticDetector(config=SemanticDetectorConfig(max_dist=3.))
    ]
    sim = Simulation(sim_dt=0.01,
                     # world_map=CirclesWorld(circles=OBSTACLES),
                     world_map=create_walls(),
                     robot_model=robot_model,
                     pedestrians_model=None,
                     sensors=sensors,
                     rt_factor=1.)
    renderer = Renderer(simulation=sim,
                        resolution=80.0,
                        screen_size=(500, 500),
                        camera="robot")
    return sim, renderer


def main():
    sim, renderer = create_sim()
    renderer.initialize()

    controller = DoMPCController(dt=0.1, goal=np.array([3., -2., 0.]), sim=sim, obstacles=OBSTACLES)
    renderer.draw("goal", CircleDrawing(controller.goal[:2], 0.1, (255, 0, 0), 0))

    running = True
    sim.step()  # First step can take some time due to Numba compilation

    u_pred = np.array([0., 0.])
    hold_time = sim.sim_dt

    while running:
        renderer.render()

        if hold_time >= controller.dt:
            semantic_reading = sim.current_state.sensors[SemanticDetector.NAME].reading
            for object_id, semantic_detection in semantic_reading.detections.items():
                print(f"{object_id}: {semantic_detection}")

            x_current = sim.current_state.world.robot.pose
            u_pred = controller.predict(x_current)
            hold_time = 0.

        start_time = time.time()
        sim.step(u_pred)
        sim.world_map.is_occupied(np.array([2.4, 2.4]))
        finish_time = time.time()
        # print(f"RT factor: {sim.sim_dt / (finish_time - start_time)}")
        hold_time += sim.sim_dt

    # Done! Time to quit.
    renderer.close()


if __name__ == '__main__':
    main()
