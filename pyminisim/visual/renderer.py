from typing import Optional, Tuple, Callable
import threading
import time

import pygame

from pyminisim.core.simulation import World
from .agents_visual import RobotVisual, PedestrianVisual


class _RendererThread(threading.Thread):
    def __init__(self, fps: float, rendering_fn: Callable):
        threading.Thread.__init__(self)
        self._sleep_time = 1.0 / fps
        self._rendering_fn = rendering_fn

    def run(self):
        while True:
            self._rendering_fn()
            time.sleep(self._sleep_time)


class Renderer:

    def __init__(self, world: World, resolution: float, screen_size: Tuple[int, int] = (500, 500), fps: float = 30.0):
        self._world = world
        self._resolution = resolution
        self._screen_size = screen_size
        self._fps = fps
        self._screen = pygame.display.set_mode(self._screen_size)
        self._robot = RobotVisual(self._resolution)
        self._pedestrians = [PedestrianVisual(self._resolution) for _ in self._world.world_state.pedestrian_poses]
        self._thread = None

    def render(self):
        state = self._world.world_state
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

    def launch(self):
        if self._thread is not None:
            return
        self._thread = _RendererThread(self._fps, lambda: self.render())
        self._thread.start()

    def stop(self):
        if self._thread is None:
            return
        self._thread.join()
