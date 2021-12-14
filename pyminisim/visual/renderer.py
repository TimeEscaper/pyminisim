from typing import Optional, Tuple, Callable
import threading
import time

import pygame

from pyminisim.core.common import Pose
from pyminisim.core.simulation import World, WorldState
from .agents_visual import RobotVisual, PedestrianVisual
from .sensors_visual import PedestrianDetectorVisual
from pyminisim.core.simulation import PedestrianDetector


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

        sensor_configs = self._world.sensor_configs
        if PedestrianDetector.NAME in sensor_configs:
            config = sensor_configs[PedestrianDetector.NAME]
            max_dist = config[PedestrianDetector.PARAM_MAX_DIST]
            fov = config[PedestrianDetector.PARAM_FOV]
            self._detector = PedestrianDetectorVisual(self._resolution, max_dist, fov)
        else:
            self._detector = None

        self._thread = None

    def render(self):
        state = self._world.world_state
        if state is not None:
            self._robot.pose = state.robot_pose
            for i, pedestrian in enumerate(self._pedestrians):
                pedestrian.pose = state.pedestrian_poses[i]

        self._screen.fill((255, 255, 255))
        self._render_robot(state)
        self._render_pedestrians(state)
        self._render_sensors(state)
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

    def _transform_pose(self, pose: Pose) -> Tuple[int, int, int]:
        return (int(pose.y * self._resolution),
                self._screen_size[1] - int(pose.x * self._resolution),
                int(pose.theta))

    def _render_robot(self, state: WorldState):
        maker = "collision" if len(state.collisions) != 0 else "normal"
        self._robot.render(self._screen,
                           self._transform_pose(state.robot_pose),
                           maker)

    def _render_pedestrians(self, state: WorldState):
        for i, pose in enumerate(state.pedestrian_poses):
            maker = "normal"
            if i in state.collisions:
                maker = "collision"
            elif PedestrianDetector.NAME in state.sensor_readings:
                if i in state.sensor_readings[PedestrianDetector.NAME]:
                    maker = "detected"
            self._pedestrians[i].render(self._screen,
                                        self._transform_pose(pose),
                                        maker)

    def _render_sensors(self, state: WorldState):
        if self._detector is not None:
            self._detector.render(self._screen, self._transform_pose(state.robot_pose))
