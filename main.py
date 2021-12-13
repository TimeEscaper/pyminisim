import time

import numpy as np
import pygame

from pyminisim.visual import Renderer
from pyminisim.core.common import Pose, Velocity
from pyminisim.core.motion import UnicycleMotion
from pyminisim.core.common import RobotAgent, PedestrianAgent
from pyminisim.core.simulation import World, PedestrianDetector


def main():
    pygame.init()

    world = World(robot=RobotAgent(Pose(3.85, 3.85, 0.0), Velocity(1.0, 10.0)),
                  pedestrians=[PedestrianAgent(Pose(7.0, 3.0, 180.0), Velocity(1.0, 10.0))],
                  sensors=[PedestrianDetector(max_dist=5.0, fov=30.0)],
                  sim_dt=0.01,
                  rt_factor=1.0)

    renderer = Renderer(world, resolution=65.0)
    renderer.launch()

    running = True
    # clock = pygame.time.Clock()
    while running:

        # Did the user click the window close button?
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        world.step()

    # Done! Time to quit.
    pygame.quit()


if __name__ == '__main__':
    main()
