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

    world = World(robot=RobotAgent(Pose(2.0, 3.85, 0.0), Velocity(0.0, 25.0)),
                  pedestrians=[PedestrianAgent(Pose(5.0, 3.85, 180.0), Velocity(0.0, 0.0))],
                  sensors=[PedestrianDetector(max_dist=3.0, fov=30.0)],
                  sim_dt=0.01,
                  rt_factor=1.0)

    renderer = Renderer(world, resolution=80.0)
    # renderer.launch()

    running = True
    # clock = pygame.time.Clock()
    while running:
        renderer.render()

        # Did the user click the window close button?
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        world.step()

    # Done! Time to quit.
    pygame.quit()


if __name__ == '__main__':
    main()
