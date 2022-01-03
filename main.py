import time
from typing import Tuple

import numpy as np
import pygame

from pyminisim.core import Simulation
from pyminisim.robot import UnicycleRobotModel
from pyminisim.pedestrians import HeadedSocialForceModelPolicy, RandomWaypointTracker
from pyminisim.sensors import PedestrianDetector
from pyminisim.visual import Renderer


def create_sim() -> Tuple[Simulation, Renderer]:
    robot_model = UnicycleRobotModel(initial_pose=np.array([2.0, 3.85, 0.0]),
                                     initial_control=np.array([0.0, np.deg2rad(25.0)]))
    tracker = RandomWaypointTracker(world_size=(7.0, 7.0))
    pedestrians_model = HeadedSocialForceModelPolicy(initial_poses=np.array([[5.0, 3.85, np.pi],
                                                                             [1.0, 1.85, -np.pi]]),
                                                     waypoint_tracker=tracker)
    sensors = [PedestrianDetector()]
    sim = Simulation(robot_model=robot_model,
                     pedestrians_model=pedestrians_model,
                     waypoint_tracker=tracker,
                     sensors=sensors)
    renderer = Renderer(simulation=sim,
                        resolution=80.0,
                        screen_size=(500, 500))
    return sim, renderer


def main():
    sim, renderer = create_sim()
    renderer.initialize()

    running = True
    sim.step()
    start_time = time.time()
    end_time = time.time()
    n_frames = 0
    while running:
        renderer.render()
        n_frames += 1

        # Did the user click the window close button?
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

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
