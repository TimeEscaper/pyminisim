from typing import Optional, Tuple

import pygame

from pyminisim.core.simulation import WorldState
from .agents_visual import RobotVisual, PedestrianVisual


class Renderer:

    def __init__(self, initial_state: WorldState, resolution: float, screen_size: Tuple[int, int] = (500, 500)):
        self._state = initial_state
        self._resolution = resolution
        self._screen_size = screen_size
        self._robot = RobotVisual(self._resolution)
        self._pedestrians = [PedestrianVisual(self._resolution) for e in self._state.pedestrian_poses]
        self._screen = pygame.display.set_mode(self._screen_size)

    def render(self, state: Optional[WorldState]):
        if state is not None:
            self._robot.pose = state.robot_pose
            for i, pedestrian in enumerate(self._pedestrians):
                pedestrian.pose = state.pedestrian_poses[i]

        self._screen.fill((255, 255, 255))
        surf, rect = self._robot.render(int(state.robot_pose.y * self._resolution),
                                        self._screen_size[1] - int(state.robot_pose.x * self._resolution),
                                        int(state.robot_pose.theta))
        self._screen.blit(surf, rect)
        for i, pose in enumerate(state.pedestrian_poses):
            surf, rect = self._pedestrians[i].render(int(pose.y * self._resolution),
                                                     self._screen_size[1] - int(pose.x * self._resolution),
                                                     int(pose.theta))
            self._screen.blit(surf, rect)
        pygame.display.flip()
