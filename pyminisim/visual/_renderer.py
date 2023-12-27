import math
from typing import Optional, Tuple, Callable
import threading
import time

import numpy as np
import pygame

from pyminisim.core import SimulationState, Simulation, PEDESTRIAN_RADIUS, ROBOT_RADIUS
from pyminisim.visual import VisualizationParams, PedestrianDetectorSkin, LidarSensorSkin, \
    AbstractDrawing, AbstractDrawingRenderer, CircleDrawing, CircleDrawingRenderer, \
    Covariance2dDrawing, Covariance2dDrawingRenderer
from pyminisim.visual.util import PoseConverter
from pyminisim.sensors import PedestrianDetector, LidarSensor
from pyminisim.world_map import EmptyWorld, CirclesWorld, AABBWorld
from ._agents import _RobotSkin, _PedestriansSkin
from ._maps import EmptyWorldSkin, CirclesWorldSkin, AABBWorldSkin


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

    CAMERA_FIXED = "fixed"
    CAMERA_ROBOT = "robot"

    _COLLISION_COLOR = (224, 54, 16)
    _GRID_COLOR = (201, 201, 201)
    _GRID_COLOR_X = (214, 51, 51)
    _GRID_COLOR_Y = (22, 179, 14)

    _GRID_AMPLITUDE = 10000

    _GRID_ARROW_ANGLE = 15.  # In degrees
    _GRID_ARROW_LENGTH = 0.2  # In metres

    def __init__(self,
                 simulation: Simulation,
                 resolution: float,
                 screen_size: Tuple[int, int] = (500, 500),
                 fps: float = 30.0,
                 grid_enabled: bool = True,
                 camera: str = "fixed"):
        camera_options = (Renderer.CAMERA_FIXED, Renderer.CAMERA_ROBOT)
        assert camera in camera_options, f"Camera must be one of the strings: {camera_options}"

        self._sim = simulation
        self._vis_params = VisualizationParams(resolution=resolution, screen_size=screen_size)
        self._pose_converter = PoseConverter(self._vis_params)
        self._fps = fps
        self._grid_enabled = grid_enabled
        self._camera = camera

        self._screen = pygame.display.set_mode(screen_size)

        self._robot = _RobotSkin(self._vis_params) if simulation.current_state.world.robot is not None else None
        self._pedestrians = _PedestriansSkin(self._vis_params) \
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
        elif isinstance(self._sim.world_map, EmptyWorld):
            self._map = EmptyWorldSkin()
        elif isinstance(self._sim.world_map, AABBWorld):
            self._map = AABBWorldSkin(self._sim.world_map, self._vis_params)
        else:
            raise ValueError(f"Unknown map type of {self._sim.world_map}")

        self._drawings = {}

        screen_half = self._vis_params.screen_size[0] // 2
        screen_limit_y = self._vis_params.screen_size[1]
        arrow_offset_sin = int(
            Renderer._GRID_ARROW_LENGTH * self._vis_params.resolution * math.sin(math.radians(self._GRID_ARROW_ANGLE)))
        arrow_offset_cos = int(
            Renderer._GRID_ARROW_LENGTH * self._vis_params.resolution * math.cos(math.radians(self._GRID_ARROW_ANGLE)))
        self._grid_arrow_x = np.array([[screen_half - arrow_offset_sin, arrow_offset_cos],
                                       [screen_half, 0],
                                       [screen_half + arrow_offset_sin, arrow_offset_cos]])
        self._grid_arrow_y = np.array([[screen_limit_y - arrow_offset_cos, screen_half - arrow_offset_sin],
                                       [screen_limit_y, screen_half],
                                       [screen_limit_y - arrow_offset_cos, screen_half + arrow_offset_sin]])
        self._grid_arrow_width = max(int(0.02 * self._vis_params.resolution), 1)

        self._thread = None

    def initialize(self):
        pygame.init()

    def render(self):
        state = self._sim.current_state
        offset = np.array([0., 0.]) if self._camera == Renderer.CAMERA_FIXED else state.world.robot.pose[:2]
        self._screen.fill((255, 255, 255))
        self._render_coordinate_grid(offset)
        self._map.render(self._screen, state, offset)
        self._render_drawings(state, offset)
        self._render_robot(state, offset)
        self._render_pedestrians(state, offset)
        self._render_sensors(state, offset)
        self._render_collisions(state, offset)
        pygame.display.flip()

    # TODO: Fix threaded renderer
    # def launch_thread(self):
    #     if self._thread is not None:
    #         return
    #     self._thread = _RendererThread(self._fps, lambda: self.render())
    #     self._thread.start()

    def close(self):
        if self._thread is not None:
            self._thread.join()
        pygame.quit()

    def draw(self, drawing_id: str, drawing: AbstractDrawing):
        if drawing.name == CircleDrawing.NAME:
            drawing_renderer = CircleDrawingRenderer(drawing, self._vis_params)
        elif drawing.name == Covariance2dDrawing.NAME:
            drawing_renderer = Covariance2dDrawingRenderer(drawing, self._vis_params)
        else:
            # TODO: Warning or exception
            return
        self._drawings[drawing_id] = drawing_renderer

    def clear_drawings(self, drawing_ids: Optional[str] = None):
        if drawing_ids is None:
            self._drawings = {}
        else:
            for drawing_name in drawing_ids:
                self._drawings.pop(drawing_name, None)

    def _render_coordinate_grid(self, offset: np.ndarray):
        if not self._grid_enabled:
            return

        resolution = self._vis_params.resolution
        for i in range(-50, 51):
            h_coordinate = self._vis_params.screen_size[0] // 2 + int((i - offset[1]) * resolution)
            v_coordinate = self._vis_params.screen_size[1] // 2 + int((i + offset[0]) * resolution)
            pygame.draw.line(self._screen,
                             color=Renderer._GRID_COLOR if i != 0 else Renderer._GRID_COLOR_X,
                             start_pos=(h_coordinate, -Renderer._GRID_AMPLITUDE),
                             end_pos=(h_coordinate, Renderer._GRID_AMPLITUDE))
            pygame.draw.line(self._screen,
                             color=Renderer._GRID_COLOR if i != 0 else Renderer._GRID_COLOR_Y,
                             start_pos=(-Renderer._GRID_AMPLITUDE, v_coordinate),
                             end_pos=(Renderer._GRID_AMPLITUDE, v_coordinate))

        pygame.draw.lines(self._screen,
                          color=Renderer._GRID_COLOR_X,
                          closed=False,
                          points=self._grid_arrow_x - np.array([int(resolution * offset[1]), 0]),
                          width=self._grid_arrow_width)
        pygame.draw.lines(self._screen,
                          color=Renderer._GRID_COLOR_Y,
                          closed=False,
                          points=self._grid_arrow_y + np.array([0, int(resolution * offset[0])]),
                          width=self._grid_arrow_width)

    def _render_robot(self, state: SimulationState, offset: np.ndarray):
        if state.world.robot is not None and self._robot is not None:
            self._robot.render(self._screen, state, offset)

    def _render_pedestrians(self, state: SimulationState, offset: np.ndarray):
        if state.world.pedestrians is not None and self._pedestrians is not None:
            self._pedestrians.render(self._screen, state, offset)

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

    def _render_sensors(self, state: SimulationState, offset: np.ndarray):
        if state.world.robot is not None:
            for sensor in self._sensors:
                sensor.render(self._screen, state, offset)

    def _render_collisions(self, state: SimulationState, offset: np.ndarray):
        robot_marker_rendered = False
        if state.world.robot is not None and state.world.robot_to_world_collision:
            self._render_marker(state.world.robot.pose, ROBOT_RADIUS, offset)
            robot_marker_rendered = True

        if state.world.robot is not None and state.world.pedestrians is not None:
            if len(state.world.robot_to_pedestrians_collisions) == 0:
                return
            if not robot_marker_rendered:
                self._render_marker(state.world.robot.pose, ROBOT_RADIUS, offset)
            for idx in state.world.robot_to_pedestrians_collisions:
                self._render_marker(state.world.pedestrians.poses[idx], PEDESTRIAN_RADIUS, offset)

    def _render_marker(self, sim_pose: np.ndarray, sim_radius: float, offset: np.ndarray):
        # TODO: Implement as another sensor
        x, y, _ = self._pose_converter.convert(sim_pose,
                                               global_offset=offset)
        pygame.draw.circle(self._screen,
                           Renderer._COLLISION_COLOR,
                           (x, y),
                           int(sim_radius * self._vis_params.resolution),
                           int(0.05 * self._vis_params.resolution))

    def _render_drawings(self, state: SimulationState, offset: np.ndarray):
        for drawing in self._drawings.values():
            drawing.render(self._screen, state, offset)
