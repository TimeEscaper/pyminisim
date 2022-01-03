import numpy as np
import pygame
from PIL import Image, ImageDraw

from pyminisim.core import SimulationState, PEDESTRIAN_RADIUS
from pyminisim.visual import AbstractSensorSkin, VisualizationParams
from pyminisim.visual.util import convert_pose
from pyminisim.sensors import PedestrianDetector, PedestrianDetectorReading, PedestrianDetectorConfig


class PedestrianDetectorSkin(AbstractSensorSkin):

    _PIE_COLOR = (95, 154, 250, 50)
    _DETECTION_COLOR = (95, 154, 250)

    _OFFSET = 90.

    def __init__(self, sensor_config: PedestrianDetectorConfig, vis_params: VisualizationParams):
        super(PedestrianDetectorSkin, self).__init__()
        self._vis_params = vis_params
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

    def render(self, screen, sim_state: SimulationState):
        self._render_detections(screen, sim_state)
        self._render_pie(screen, sim_state)

    def _render_detections(self, screen, sim_state: SimulationState):
        reading = sim_state.sensors[PedestrianDetector.NAME]
        assert isinstance(reading, PedestrianDetectorReading)
        if len(reading.pedestrians) == 0:
            return
        detected_poses = sim_state.world.pedestrians_poses[list(reading.pedestrians.keys())]
        pixel_poses = convert_pose(detected_poses, self._vis_params)
        for pixel_pose in pixel_poses:
            x, y, _ = pixel_pose
            pygame.draw.circle(screen,
                               PedestrianDetectorSkin._DETECTION_COLOR,
                               (x, y),
                               int(PEDESTRIAN_RADIUS * self._vis_params.resolution),
                               int(0.05 * self._vis_params.resolution))

    def _render_pie(self, screen, sim_state: SimulationState):
        x, y, theta = convert_pose(sim_state.world.robot_pose, self._vis_params,
                                   PedestrianDetectorSkin._OFFSET + self._fov / 2.)
        surf = pygame.transform.rotate(self._surf, theta)
        rect = surf.get_rect(center=(x, y))
        screen.blit(surf, rect)
