from typing import List, Dict, Tuple
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
                 sim_dt: float):
        self._sensors = sensors
        self._sim_dt = sim_dt
        self._sim = Simulation(robot, pedestrians, sim_dt)
        self._world_state = self._get_world_state()

    def step(self):
        self._sim.step()
        time.sleep(self._sim_dt)
        self._world_state = self._get_world_state()

    @property
    def world_state(self) -> WorldState:
        return self._world_state

    def _get_world_state(self) -> WorldState:
        sensor_readings = {sensor.name: sensor.get_reading(self._sim) for sensor in self._sensors}
        robot_pose = Pose(self._sim.robot_pose[0], self._sim.robot_pose[1], self._sim.robot_pose[2])
        pedestrian_poses = [Pose(e[0], e[1], e[2]) for e in self._sim.pedestrians_poses]
        collisions = [i for i, e in enumerate(self._sim.pedestrians_poses)
                      if np.linalg.norm(self._sim.robot_pose - e) < (Simulation.ROBOT_RADIUS +
                                                                     Simulation.PEDESTRIAN_RADIUS)]
        return WorldState(sensor_readings, robot_pose, pedestrian_poses, collisions)
