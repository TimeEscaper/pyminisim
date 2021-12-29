from typing import List, Dict, Optional
import time
from dataclasses import dataclass

import numpy as np

from pyminisim.core.simulation import Simulation
from pyminisim.core.simulation import AbstractSensor
from pyminisim.core.common import RobotAgent, PedestrianAgent, Pose


@dataclass
class WorldState:
    sensor_readings: Dict
    robot_pose: Pose
    pedestrian_poses: List[Pose]
    collisions: List[int]


class World:

    def __init__(self,
                 robot: RobotAgent,
                 pedestrians: List[PedestrianAgent],
                 sensors: List[AbstractSensor],
                 sim_dt: float,
                 rt_factor: Optional[float] = 1.0):
        self._sensors = sensors
        self._sim_dt = sim_dt
        self._rt_factor = rt_factor
        self._sim = Simulation(robot, pedestrians, sim_dt)
        self._world_state = self._get_world_state()

    def step(self):
        time_start = time.time()
        self._sim.step()
        time_end = time.time()
        time_elapsed = time_end - time_start
        if self._rt_factor is not None:
            time_to_sleep = self._sim_dt / self._rt_factor - time_elapsed
            if time_to_sleep > 0.:
                # TODO: Log think on logging RT factor info
                time.sleep(self._sim_dt / self._rt_factor)
        self._world_state = self._get_world_state()

    @property
    def world_state(self) -> WorldState:
        return self._world_state

    @property
    def sensor_configs(self) -> Dict:
        return {sensor.name: sensor.get_config() for sensor in self._sensors}

    def _get_world_state(self) -> WorldState:
        sensor_readings = {sensor.name: sensor.get_reading(self._sim) for sensor in self._sensors}
        robot_pose = Pose(self._sim.robot_pose[0], self._sim.robot_pose[1], self._sim.robot_pose[2])
        pedestrian_poses = [Pose(e[0], e[1], e[2]) for e in self._sim.pedestrians_poses]
        collisions = [i for i, e in enumerate(self._sim.pedestrians_poses)
                      if np.linalg.norm(self._sim.robot_pose[:2] - e[:2]) < (Simulation.ROBOT_RADIUS +
                                                                             Simulation.PEDESTRIAN_RADIUS)]
        return WorldState(sensor_readings, robot_pose, pedestrian_poses, collisions)
