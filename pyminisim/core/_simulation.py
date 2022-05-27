import time
from typing import Optional, List, Dict

import numpy as np

from ._constants import ROBOT_RADIUS, PEDESTRIAN_RADIUS
from ._motion import AbstractRobotMotionModel
from ._pedestrians_model import AbstractPedestriansModel
from ._sensor import AbstractSensor
from ._simulation_state import SimulationState
# from pyminisim.core import AbstractRobotMotionModel, AbstractPedestriansPolicy, AbstractWaypointTracker, \
#     AbstractSensorReading, AbstractSensor, ROBOT_RADIUS, PEDESTRIAN_RADIUS
from ._world_state import WorldState
from ._world_map import AbstractWorldMap


class Simulation:

    def __init__(self,
                 world_map: AbstractWorldMap,
                 robot_model: Optional[AbstractRobotMotionModel],
                 pedestrians_model: Optional[AbstractPedestriansModel],
                 sensors: List[AbstractSensor],
                 sim_dt: float = 0.01,
                 rt_factor: Optional[float] = 1.0):
        self._world_map = world_map
        self._robot_model = robot_model
        self._pedestrians_model = pedestrians_model
        self._sensors = sensors
        self._sim_dt = sim_dt
        self._rt_factor = rt_factor

        self._backup_robot_model = None
        self._backup_pedestrians_model = None

        self._current_state = self._get_simulation_state()

    @property
    def world_map(self) -> AbstractWorldMap:
        return self._world_map

    @property
    def current_state(self) -> SimulationState:
        return self._current_state

    @property
    def sensors(self) -> List[AbstractSensor]:
        return self._sensors

    def step(self, control: Optional[np.ndarray] = None) -> SimulationState:
        time_start = time.time()
        self._make_steps(control)
        time_end = time.time()
        time_elapsed = time_end - time_start
        if self._rt_factor is not None:
            time_to_sleep = self._sim_dt / self._rt_factor - time_elapsed
            if time_to_sleep > 0.:
                # TODO: Log think on logging RT factor info
                time.sleep(self._sim_dt / self._rt_factor)

        self._current_state = self._get_simulation_state()

        return self._current_state

    def reset_to_state(self, state: WorldState):
        if self._robot_model is not None:
            self._robot_model.reset_to_state(state.robot)
        if self._pedestrians_model is not None:
            self._pedestrians_model.reset_to_state(state.pedestrians)
        self._current_state = self._get_simulation_state()

    def set_robot_enabled(self, enabled: bool):
        if enabled:
            if self._robot_model is not None:
                return
            if self._backup_robot_model is None:
                raise RuntimeError("Robot model was not initialized")
            self._robot_model = self._backup_robot_model
            self._backup_robot_model = None
        else:
            if self._backup_robot_model is not None:
                return
            if self._robot_model is None:
                raise RuntimeError("Robot model was not initialized")
            self._backup_robot_model = self._robot_model
            self._robot_model = None
        self._current_state = self._get_simulation_state()

    def set_pedestrians_enabled(self, enabled: bool):
        if enabled:
            if self._pedestrians_model is not None:
                return
            if self._backup_pedestrians_model is None:
                raise RuntimeError("Pedestrians model was not initialized")
            self._pedestrians_model = self._backup_pedestrians_model
            self._backup_pedestrians_model = None
        else:
            if self._backup_pedestrians_model is not None:
                return
            if self._pedestrians_model is None:
                raise RuntimeError("Pedestrians model was not initialized")
            self._backup_pedestrians_model = self._pedestrians_model
            self._pedestrians_model = None
        self._current_state = self._get_simulation_state()

    def _make_steps(self, control: Optional[np.ndarray]):
        if self._robot_model is not None:
            self._robot_model.step(self._sim_dt, control)
            robot_pose = self._robot_model.state.pose
            robot_velocity = self._robot_model.state.velocity
        else:
            robot_pose = None
            robot_velocity = None

        if self._pedestrians_model is not None:
            self._pedestrians_model.step(self._sim_dt, robot_pose, robot_velocity)

    def _get_simulation_state(self) -> SimulationState:
        world_state = self._get_world_state()
        sensors_readings = self._get_sensors_readings(world_state, self._world_map)
        return SimulationState(world=world_state, sensors=sensors_readings)

    def _get_world_state(self) -> WorldState:
        if self._robot_model is not None:
            if self._pedestrians_model is not None:
                collisions = [i for i, e in enumerate(self._pedestrians_model.state.poses)
                              if np.linalg.norm(self._robot_model.state.pose[:2] - e[:2])
                              < (ROBOT_RADIUS + PEDESTRIAN_RADIUS)]
            else:
                collisions = None
            robot_state = self._robot_model.state
        else:
            collisions = None
            robot_state = None

        if self._pedestrians_model is not None:
            pedestrians_state = self._pedestrians_model.state
        else:
            pedestrians_state = None

        return WorldState(robot=robot_state,
                          pedestrians=pedestrians_state,
                          robot_to_pedestrians_collisions=collisions)

    def _get_sensors_readings(self, world_state: WorldState, world_map: AbstractWorldMap) -> Dict:
        if self._robot_model is not None:
            return {sensor.sensor_name: sensor.get_reading(world_state, world_map) for sensor in self._sensors}
        else:
            return {}
