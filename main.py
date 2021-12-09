import time

import numpy as np
import pygame

from pyminisim.visual.agents_visual import RobotVisual, PedestrianVisual
from pyminisim.core.common import Pose
from pyminisim.core.motion import UnicycleMotion


def main():
    pygame.init()

    # Set up the drawing window
    screen = pygame.display.set_mode([500, 500])

    scale = 65.0  # pixels per meter
    motion_model = UnicycleMotion(initial_poses=np.array([[0.0, 0.0, 0.0]]),
                                  initial_velocities=np.array([[1.0, 10.0]]))
    sim_dt = 0.04

    # Run until the user asks to quit
    running = True
    while running:

        # Did the user click the window close button?
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # Fill the background with white
        screen.fill((255, 255, 255))

        # Draw a solid blue circle in the center
        # pygame.draw.circle(screen, (0, 0, 255), (250, 250), 75)
        robot_pose = motion_model.poses[0]
        robot_pose = Pose(int(robot_pose[0] * scale) + 250,
                          int(robot_pose[1] * scale) + 250,
                          robot_pose[2])
        robot = RobotVisual(robot_pose)
        screen.blit(robot.surf, robot.rect)

        # Flip the display
        pygame.display.flip()

        motion_model.step(sim_dt)
        time.sleep(sim_dt)

    # Done! Time to quit.
    pygame.quit()


if __name__ == '__main__':
    main()
