import sys

sys.path.append('..')

import time
from typing import Tuple

import numpy as np
import pygame

from pyminisim.core import Simulation
from pyminisim.world_map import EmptyWorld, CirclesWorld
from pyminisim.robot import UnicycleRobotModel
from pyminisim.pedestrians import HeadedSocialForceModelPolicy, RandomWaypointTracker
from pyminisim.sensors import PedestrianDetectorNoise, PedestrianDetector, OmniObstacleDetector
from pyminisim.visual import Renderer, CircleDrawing


class MPCController:

    def __init__(self, dt: float):
        self._dt = dt

    def _forward_dynamics(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        return np.array([x[0] + u[0] * np.cos(x[2]) * self._dt,
                         x[1] + u[0] * np.sin(x[2]) * self._dt,
                         MPCController._wrap_angle(x[2] + u[1] * self._dt)])

    def _get_dynamics_matrices(self, x: np.ndarray, u: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        A = np.eye(3)
        B = np.array([[np.cos(x[2]) * self._dt, 0.],
                      [np.sin(x[2]) * self._dt, 0.],
                      [0., self._dt]])
        return A, B

    @staticmethod
    def _wrap_angle(theta: float) -> float:
        return (theta + np.pi) % (2 * np.pi) - np.pi


def create_sim() -> Tuple[Simulation, Renderer]:
    robot_model = UnicycleRobotModel(initial_pose=np.array([2.0, 3.85, 0.0]),
                                     initial_control=np.array([0.0, np.deg2rad(25.0)]))
    tracker = RandomWaypointTracker(world_size=(7.0, 7.0))
    pedestrians_model = HeadedSocialForceModelPolicy(n_pedestrians=5,
                                                     waypoint_tracker=tracker)
    # You can model sensor's noise
    # sensor_noise = PedestrianDetectorNoise(distance_mu=0., distance_sigma=0.2,
    #                                        angle_mu=0., angle_sigma=0.05,
    #                                        misdetection_prob=0.1)
    sensor_noise = None
    sensors = [PedestrianDetector(noise=sensor_noise), OmniObstacleDetector()]
    sim = Simulation(world_map=EmptyWorld(), # CirclesWorld(circles=np.array([[3, 3, 1]])),
                     robot_model=robot_model,
                     pedestrians_model=pedestrians_model,
                     sensors=sensors)
    renderer = Renderer(simulation=sim,
                        resolution=80.0,
                        screen_size=(500, 500))
    return sim, renderer


def main():
    sim, renderer = create_sim()
    renderer.initialize()

    running = True
    sim.step()  # First step can take some time due to Numba compilation
    start_time = time.time()
    end_time = time.time()
    n_frames = 0
    while running:
        renderer.render()
        n_frames += 1
        sim.step()
        current_time = time.time()
        if current_time - start_time >= 20.0:
            end_time = current_time
            break
    print("FPS: ", n_frames / (end_time - start_time))

    # Done! Time to quit.
    pygame.quit()


if __name__ == '__main__':
    main()
