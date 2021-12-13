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

    # Set up the drawing window
    screen = pygame.display.set_mode([500, 500])

    world = World(robot=RobotAgent(Pose(3.0, 3.0, 0.0), Velocity(0.0, 10.0)),
                  pedestrians=[PedestrianAgent(Pose(7.0, 3.0, 180.0))],
                  sensors=[PedestrianDetector(max_dist=5.0, fov=30.0)],
                  sim_dt=0.01)

    renderer = Renderer(world.world_state, resolution=65.0)

    running = True
    # clock = pygame.time.Clock()
    while running:

        # Did the user click the window close button?
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        world.step()
        renderer.render(world.world_state)

    # Done! Time to quit.
    pygame.quit()


if __name__ == '__main__':
    main()
