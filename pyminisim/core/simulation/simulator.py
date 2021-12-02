import numpy as np

from pyminisim.core.motion import UnicycleMotion


class Simulation:

    def __init__(self,
                 pedestrians_poses: np.ndarray,
                 pedestrians_velocities: np.ndarray,
                 dt: float):
        self._motion = UnicycleMotion(pedestrians_poses)
        self._dt = dt

    @property
    def current_pedestrians_poses(self) -> np.ndarray:
        return self._motion.poses

    def step(self):
        self._motion.simulate(self._dt)
