import math
from abc import ABC
from typing import Tuple
import pkg_resources

import pygame
import numpy as np

from pyminisim.core import SimulationState, PEDESTRIAN_RADIUS
from pyminisim.visual import VisualizationParams
from pyminisim.visual.util import PoseConverter
from pyminisim.util import wrap_angle


class _AbstractAgentSprite(pygame.sprite.Sprite, ABC):

    def __init__(self,
                 asset: str,
                 radius: float,
                 resolution: float):
        super(_AbstractAgentSprite, self).__init__()
        square_side = int(2 * radius * resolution / math.sqrt(2))
        self._surf = pygame.image.load(asset).convert_alpha()
        self._surf = pygame.transform.scale(self._surf, (square_side, square_side))

    def render(self, screen, pixel_pose: Tuple[int, int, int]):
        x, y, theta = pixel_pose
        rect = self._surf.get_rect(center=(x, y))
        surf = pygame.transform.rotate(self._surf, theta)
        rect = surf.get_rect(center=rect.center)
        screen.blit(surf, rect)


class _RobotSprite(_AbstractAgentSprite):

    _ASSET_PACKAGE = "pyminisim.visual"
    _ASSET_PATH = "assets/robot_3Dred.png"

    def __init__(self, robot_radius: float, resolution: float):
        asset_full_path = pkg_resources.resource_filename(_RobotSprite._ASSET_PACKAGE,
                                                          _RobotSprite._ASSET_PATH)
        super(_RobotSprite, self).__init__(asset_full_path, robot_radius, resolution)


class _PedestrianSprite(_AbstractAgentSprite):

    _ASSET_PACKAGE = "pyminisim.visual"
    _ASSET_PATH = "assets/character1.png"

    def __init__(self, resolution: float):
        asset_full_path = pkg_resources.resource_filename(_PedestrianSprite._ASSET_PACKAGE,
                                                          _PedestrianSprite._ASSET_PATH)
        super(_PedestrianSprite, self).__init__(asset_full_path, PEDESTRIAN_RADIUS, resolution)


class _PedestriansSkin:

    _OFFSET = 180.

    def __init__(self, vis_params: VisualizationParams):
        self._pose_converter = PoseConverter(vis_params)
        self._resolution = vis_params.resolution
        self._sprites = None

    def render(self, screen, sim_state: SimulationState, global_offset: np.ndarray):
        ped_poses = np.stack(list(sim_state.world.pedestrians.poses.values()), axis=0)
        pixel_poses = self._pose_converter.convert(ped_poses,
                                                   global_offset=global_offset,
                                                   angle_offset_degrees=_PedestriansSkin._OFFSET)
        if self._sprites is None or len(pixel_poses) != len(self._sprites):
            self._sprites = [_PedestrianSprite(self._resolution) for _ in range(len(pixel_poses))]
        for pixel_pose, sprite in zip(pixel_poses, self._sprites):
            sprite.render(screen, pixel_pose)


class _UnicycleRobotSkin:

    _OFFSET = 90.

    def __init__(self, robot_radius: float, vis_params: VisualizationParams):
        self._pose_converter = PoseConverter(vis_params)
        self._sprite = _RobotSprite(robot_radius, vis_params.resolution)

    def render(self, screen, sim_state: SimulationState, global_offset: np.ndarray):
        self._sprite.render(screen, self._pose_converter.convert(sim_state.world.robot.pose,
                                                                 global_offset=global_offset,
                                                                 angle_offset_degrees=_UnicycleRobotSkin._OFFSET))


class _BicycleRobotSkin:

    _CIRCLE_WIDTH = 0.05
    _CIRCLE_COLOR = (150, 146, 146)
    _WHEEL_COLOR = (0, 0, 0)
    _OFFSET = 90.

    def __init__(self,
                 robot_radius: float,
                 wheel_base: float,
                 vis_params: VisualizationParams) -> None:
        self._robot_radius = robot_radius
        self._wheel_base = wheel_base
        self._pose_converter = PoseConverter(vis_params)
        self._vis_params = vis_params
        
        # Calculate wheel dimensions automatically based on wheel_base and robot_radius
        # Wheel length: proportional to robot size, fits nicely within the body
        self._wheel_length = min(wheel_base / 3.0, robot_radius * 0.7)
        # Wheel width: typical wheel aspect ratio (~1:3)
        self._wheel_width = self._wheel_length / 3.0

    def _draw_wheel(self, screen, pixel_pose: Tuple[int, int, int]):
        """Draw a rotated rectangle representing a wheel at the given pixel pose."""
        cx, cy, theta_deg = pixel_pose
        theta = np.deg2rad(-(theta_deg + 90))  # Negate for pygame's coordinate system
        
        # Half dimensions in pixels
        hl = self._wheel_length * self._vis_params.resolution / 2.0  # half length
        hw = self._wheel_width * self._vis_params.resolution / 2.0   # half width
        
        # Rotation matrix components
        cos_t = np.cos(theta)
        sin_t = np.sin(theta)
        
        # Calculate the four corners of the rotated rectangle
        corners = [
            (cx + hl * cos_t - hw * sin_t, cy + hl * sin_t + hw * cos_t),
            (cx + hl * cos_t + hw * sin_t, cy + hl * sin_t - hw * cos_t),
            (cx - hl * cos_t + hw * sin_t, cy - hl * sin_t - hw * cos_t),
            (cx - hl * cos_t - hw * sin_t, cy - hl * sin_t + hw * cos_t),
        ]
        
        pygame.draw.polygon(screen, _BicycleRobotSkin._WHEEL_COLOR, corners)

    def render(self, screen, sim_state: SimulationState, global_offset: np.ndarray):
        pose_center = sim_state.world.robot.pose
        x_center = pose_center[0]
        y_center = pose_center[1]
        theta = pose_center[2]
        delta = sim_state.world.robot.control[1]
        d = self._wheel_base / 2.

        pose_rear = np.array([
            x_center - d * np.cos(theta),
            y_center - d * np.sin(theta),
            theta
        ])
        pose_front = np.array([
            x_center + d * np.cos(theta),
            y_center + d * np.sin(theta),
            wrap_angle(theta + delta)  # Changed back to + for correct direction with negated angle
        ])

        pixels_center = self._pose_converter.convert(pose_center,
                                                     global_offset=global_offset)
        pixels_rear = self._pose_converter.convert(pose_rear,
                                                   global_offset=global_offset)
        pixels_front = self._pose_converter.convert(pose_front,
                                                    global_offset=global_offset)

        # Draw the body circle
        pygame.draw.circle(screen,
                           _BicycleRobotSkin._CIRCLE_COLOR,
                           (pixels_center[0], pixels_center[1]),
                           int(self._robot_radius * self._vis_params.resolution),
                           int(_BicycleRobotSkin._CIRCLE_WIDTH * self._vis_params.resolution))

        # Draw rear wheel
        self._draw_wheel(screen, pixels_rear)
        
        # Draw front wheel (steerable)
        self._draw_wheel(screen, pixels_front)


