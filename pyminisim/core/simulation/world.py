from typing import List, Dict
import time

import numpy as np

from pyminisim.core.simulation import Simulation
from pyminisim.core.simulation import AbstractSensor
from pyminisim.core.common import RobotAgent, PedestrianAgent


class World:

    def __init__(self,
                 robot: RobotAgent,
                 pedestrians: List[PedestrianAgent],
                 sensors: List[AbstractSensor],
                 sim_dt: float):
        self._sensors = sensors
        self._sim_dt = sim_dt
        self._sim = Simulation(robot, pedestrians, sim_dt)

    def step(self):
        self._sim.step()
        time.sleep(self._sim_dt)

    def get_sensor_readings(self) -> Dict:
        return {sensor.name: sensor.get_reading(self._sim) for sensor in self._sensors}
