from typing import Optional, Tuple, Callable
import threading
import time

import numpy as np
import pygame

from pyminisim.core import SimulationState, Simulation, PEDESTRIAN_RADIUS, ROBOT_RADIUS
from pyminisim.visual import VisualizationParams, PedestrianDetectorSkin, LidarSensorSkin, \
    AbstractDrawing, AbstractDrawingRenderer, CircleDrawing, CircleDrawingRenderer
from pyminisim.visual.util import PoseConverter
from pyminisim.sensors import PedestrianDetector, LidarSensor
from pyminisim.world_map import EmptyWorld, CirclesWorld
from ._agents import _RobotSkin, _PedestriansSkin
from ._maps import EmptyWorldSkin, CirclesWorldSkin


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

    _COLLISION_COLOR = (224, 54, 16)

    def __init__(self,
                 simulation: Simulation,
                 resolution: float,
                 screen_size: Tuple[int, int] = (500, 500),
                 fps: float = 30.0):
        self._sim = simulation
        self._vis_params = VisualizationParams(resolution=resolution, screen_size=screen_size)
        self._pose_converter = PoseConverter(self._vis_params)
        self._fps = fps

        self._screen = pygame.display.set_mode(screen_size)

        self._robot = _RobotSkin(self._vis_params) if simulation.current_state.world.robot is not None else None
        self._pedestrians = _PedestriansSkin(
            simulation.current_state.world.pedestrians.poses.shape[0], self._vis_params) \
            if simulation.current_state.world.pedestrians is not None else None

        # TODO: Decouple sensors visualization
        self._sensors = []
        for sensor in self._sim.sensors:
            if sensor.sensor_name == PedestrianDetector.NAME:
                self._sensors.append(PedestrianDetectorSkin(sensor.sensor_config, self._vis_params))
            elif sensor.sensor_name == LidarSensor.NAME:
                self._sensors.append(LidarSensorSkin(sensor.sensor_config, self._vis_params))

        # TODO: Decouple world map visualization
        if isinstance(self._sim.world_map, CirclesWorld):
            self._map = CirclesWorldSkin(self._sim.world_map, self._vis_params)
        else:
            self._map = EmptyWorldSkin()

        self._drawings = {}

        self._thread = None

    def initialize(self):
        pygame.init()

    def render(self):
        state = self._sim.current_state
        self._screen.fill((255, 255, 255))
        self._map.render(self._screen, state)
        self._render_drawings(state)
        self._render_robot(state)
        self._render_pedestrians(state)
        self._render_sensors(state)
        self._render_collisions(state)
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

    def draw(self, id: str, drawing: AbstractDrawing):
        if drawing.name == CircleDrawing.NAME:
            drawing_renderer = CircleDrawingRenderer(drawing, self._vis_params)
        else:
            # TODO: Warning or exception
            return
        self._drawings[id] = drawing_renderer

    def clear_drawings(self, drawing_ids: Optional[str] = None):
        if drawing_ids is None:
            self._drawings = {}
        else:
            for drawing_name in drawing_ids:
                self._drawings.pop(drawing_name, None)

    def _render_robot(self, state: SimulationState):
        if state.world.robot is not None and self._robot is not None:
            self._robot.render(self._screen, state)

    def _render_pedestrians(self, state: SimulationState):
        if state.world.pedestrians is not None and self._pedestrians is not None:
            self._pedestrians.render(self._screen, state)

    # def _render_pedestrians(self, state: WorldState):
    #     for i, pose in enumerate(state.pedestrian_poses):
    #         maker = "normal"
    #         if i in state.collisions:
    #             maker = "collision"
    #         elif PedestrianDetector.NAME in state.sensor_readings:
    #             if i in state.sensor_readings[PedestrianDetector.NAME]:
    #                 maker = "detected"
    #         self._pedestrians[i].render(self._screen,
    #                                     self._transform_pose(pose),
    #                                     maker)

    def _render_sensors(self, state: SimulationState):
        if state.world.robot is not None:
            for sensor in self._sensors:
                sensor.render(self._screen, state)

    def _render_collisions(self, state: SimulationState):
        if state.world.robot is not None and state.world.pedestrians is not None:
            if len(state.world.robot_to_pedestrians_collisions) == 0:
                return
            self._render_marker(state.world.robot.pose, ROBOT_RADIUS)
            for idx in state.world.robot_to_pedestrians_collisions:
                self._render_marker(state.world.pedestrians.poses[idx], PEDESTRIAN_RADIUS)

    def _render_marker(self, sim_pose: np.ndarray, sim_radius: float):
        # TODO: Implement as another sensor
        x, y, _ = self._pose_converter.convert(sim_pose)
        pygame.draw.circle(self._screen,
                           Renderer._COLLISION_COLOR,
                           (x, y),
                           int(sim_radius * self._vis_params.resolution),
                           int(0.05 * self._vis_params.resolution))

    def _render_drawings(self, state: SimulationState):
        for drawing in self._drawings.values():
            drawing.render(self._screen, state)
