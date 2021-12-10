import time

import numpy as np
import pygame

from pyminisim.visual.agents_visual import RobotVisual, PedestrianVisual
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

    resolution = 65.0  # pixels per meter

    # Run until the user asks to quit
    running = True
    # clock = pygame.time.Clock()
    while running:

        # Did the user click the window close button?
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # Fill the background with white
        screen.fill((255, 255, 255))

        # Draw a solid blue circle in the center
        # pygame.draw.circle(screen, (0, 0, 255), (250, 250), 75)
        robot_pose = world._sim.robot_pose
        robot_pose = Pose(robot_pose[0],
                          robot_pose[1],
                          robot_pose[2])
        robot = RobotVisual(robot_pose, resolution)
        screen.blit(robot.surf, robot.rect)

        for ped in world._sim.pedestrians_poses:
            ped_pose = Pose(ped[0],
                            ped[1],
                            ped[2])
            ped_vis = PedestrianVisual(ped_pose, resolution)
            screen.blit(ped_vis.surf, ped_vis.rect)

        # Flip the display
        pygame.display.flip()

        world.step()
        readings = world.get_sensor_readings()["pedestrian_detector"]
        if len(readings) != 0:
            print(readings)
        # clock.tick(30)

    # Done! Time to quit.
    pygame.quit()


if __name__ == '__main__':
    main()
