import sys

sys.path.append('..')

import time
from typing import Tuple

import numpy as np
import pygame

import pyminisim as pms


def create_sim() -> Tuple[pms.core.Simulation, pms.visual.Renderer]:
    robot_model = pms.robot.UnicycleRobotModel(initial_pose=np.array([0., 0., 0.]),
                                               initial_control=np.array([0., np.deg2rad(0.)]))
    sensors = [pms.sensors.LidarSensor()]
    world = pms.world_map.LinesWorld(lines=np.array([[[2., -2],
                                                      [2., 2]],
                                                     [[-2, -2],
                                                      [-2, 2]]]))
    # world = pms.world_map.LinesWorld(lines=np.array([[[2., 0.],
    #                                                   [2., 3.]],
    #                                                  [[2., 3.],
    #                                                   [0., 3.]]]),
    #                                  line_width=0.1)
    # world = pms.world_map.CirclesWorld(np.array([[2., 2., 0.15]]))
    sim = pms.core.Simulation(sim_dt=0.01,
                              world_map=world,
                              robot_model=robot_model,
                              pedestrians_model=None,
                              sensors=[])
    renderer = pms.visual.Renderer(simulation=sim,
                                   resolution=80.0,
                                   screen_size=(500, 500))

    p1 = np.array([3., 1.])
    p2, _ = world.closest_point(p1)
    renderer.draw("p1", pms.visual.CircleDrawing(p1, 0.04, (255, 0, 0)))
    renderer.draw("p2", pms.visual.CircleDrawing(p2, 0.04, (255, 0, 0)))

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
        if current_time - start_time >= 1000.0:
            end_time = current_time
            break
    print("FPS: ", n_frames / (end_time - start_time))

    # Done! Time to quit.
    pygame.quit()


if __name__ == '__main__':
    main()
