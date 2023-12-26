import numpy as np
import pygame
from PIL import Image, ImageDraw

from pyminisim.core import SimulationState, PEDESTRIAN_RADIUS
from pyminisim.visual import AbstractSensorSkin, VisualizationParams
from pyminisim.visual.util import PoseConverter
from pyminisim.sensors import PedestrianDetector, PedestrianDetectorReading, PedestrianDetectorConfig, \
    LidarSensor, LidarSensorReading, LidarSensorConfig


class PedestrianDetectorSkin(AbstractSensorSkin):
    _PIE_COLOR = (95, 154, 250, 50)
    _DETECTION_COLOR = (95, 154, 250)

    _OFFSET = 90.

    def __init__(self, sensor_config: PedestrianDetectorConfig, vis_params: VisualizationParams):
        # TODO: Config type assertions
        super(PedestrianDetectorSkin, self).__init__()
        self._vis_params = vis_params
        self._pose_converter = PoseConverter(vis_params)
        self._fov = np.rad2deg(sensor_config.fov)
        max_dist = sensor_config.max_dist
        r = int(vis_params.resolution * max_dist)
        pil_image = Image.new("RGBA", (2 * r, 2 * r))
        pil_draw = ImageDraw.Draw(pil_image)
        pil_draw.pieslice((0, 0, 2 * r, 2 * r), 0, self._fov, fill=PedestrianDetectorSkin._PIE_COLOR)
        mode = pil_image.mode
        size = pil_image.size
        data = pil_image.tobytes()
        self._surf = pygame.image.fromstring(data, size, mode)

    def render(self, screen, sim_state: SimulationState, global_offset: np.ndarray):
        self._render_detections(screen, sim_state, global_offset)
        self._render_pie(screen, sim_state, global_offset)

    def _render_detections(self, screen, sim_state: SimulationState, global_offset: np.ndarray):
        reading = sim_state.sensors[PedestrianDetector.NAME].reading
        assert isinstance(reading, PedestrianDetectorReading)
        if len(reading.pedestrians) == 0:
            return
        detected_poses = np.array([sim_state.world.pedestrians.poses[k]
                                   for k in reading.pedestrians.keys()])
        pixel_poses = self._pose_converter.convert(detected_poses, global_offset=global_offset)
        for pixel_pose in pixel_poses:
            x, y, _ = pixel_pose
            pygame.draw.circle(screen,
                               PedestrianDetectorSkin._DETECTION_COLOR,
                               (x, y),
                               int(PEDESTRIAN_RADIUS * self._vis_params.resolution),
                               int(0.05 * self._vis_params.resolution))

    def _render_pie(self, screen, sim_state: SimulationState, global_offset: np.ndarray):
        x, y, theta = self._pose_converter.convert(sim_state.world.robot.pose,
                                                   global_offset=global_offset,
                                                   angle_offset_degrees=PedestrianDetectorSkin._OFFSET + self._fov / 2.)
        surf = pygame.transform.rotate(self._surf, theta)
        rect = surf.get_rect(center=(x, y))
        screen.blit(surf, rect)


class LidarSensorSkin(AbstractSensorSkin):
    _POINT_COLOR = (255, 0, 0)
    _POINT_RADIUS = 0.03

    def __init__(self, sensor_config: LidarSensorConfig, vis_params: VisualizationParams):
        super(LidarSensorSkin, self).__init__()
        self._pose_converter = PoseConverter(vis_params)
        self._pixel_radius = int(LidarSensorSkin._POINT_RADIUS * vis_params.resolution)

    def render(self, screen, sim_state: SimulationState, global_offset: np.ndarray):
        if LidarSensor.NAME not in sim_state.sensors:
            return
        reading = sim_state.sensors[LidarSensor.NAME].reading
        if not isinstance(reading, LidarSensorReading):
            return
        if len(reading.points) == 0:
            return

        pixel_points = self._pose_converter.convert(reading.points, global_offset=global_offset)
        for pixel_point in pixel_points:
            x, y = pixel_point
            pygame.draw.circle(screen,
                               LidarSensorSkin._POINT_COLOR,
                               (x, y),
                               self._pixel_radius,
                               0)
