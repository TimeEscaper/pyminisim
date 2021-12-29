import time

import pygame

from pyminisim.core.common import Pose, Velocity
from pyminisim.core.common import RobotAgent, PedestrianAgent
from pyminisim.core.simulation import World, PedestrianDetector
from pyminisim.visual import Renderer


def main():
    pygame.init()

    step = 5.0
    # pedestrians = [PedestrianAgent(Pose(i * step, 3.85, 180.0), Velocity(0.0, 0.0)) for i in range(100)]
    world = World(robot=RobotAgent(Pose(2.0, 3.85, 0.0), Velocity(0.0, 25.0)),
                  pedestrians=[PedestrianAgent(Pose(5.0, 3.85, 180.0), Velocity(0.0, 0.0))],
                  sensors=[PedestrianDetector(max_dist=3.0, fov=30.0)],
                  sim_dt=0.01,
                  rt_factor=1.0)

    renderer = Renderer(world, resolution=80.0)
    # renderer.launch()

    running = True
    # clock = pygame.time.Clock()
    world.step()
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

        world.step()
        current_time = time.time()
        if current_time - start_time >= 20.0:
            end_time = current_time
            break
    print("FPS: ", n_frames / (end_time - start_time))

    # Done! Time to quit.
    pygame.quit()


if __name__ == '__main__':
    main()
