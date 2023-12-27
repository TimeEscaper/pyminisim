import time
import random
from typing import Optional, List, Dict

import numpy as np

from ._constants import ROBOT_RADIUS, PEDESTRIAN_RADIUS
from ._motion import AbstractRobotMotionModel
from ._pedestrians_model import AbstractPedestriansModel
from ._sensor import AbstractSensor, SensorState
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
                 rt_factor: Optional[float] = None):
        self._world_map = world_map
        self._robot_model = robot_model
        self._pedestrians_model = pedestrians_model
        self._sensors = sensors
        self._sim_dt = sim_dt
        self._rt_factor = rt_factor

        self._backup_robot_model = None
        self._backup_pedestrians_model = None

        self._robot_to_world_collision = False

        self._current_state = self._get_simulation_state({})

    @property
    def sim_dt(self) -> float:
        return self._sim_dt

    @property
    def world_map(self) -> AbstractWorldMap:
        return self._world_map

    @property
    def current_state(self) -> SimulationState:
        return self._current_state

    @property
    def sensors(self) -> List[AbstractSensor]:
        return self._sensors

    @staticmethod
    def seed(seed: int) -> None:
        random.seed(seed)
        np.random.seed(seed)

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

        self._current_state = self._get_simulation_state(self._current_state.sensors)

        return self._current_state

    def reset_to_state(self, state: WorldState):
        # TODO: Add sensors state to argument
        self._world_map.reset_to_state(state.world_map)
        if self._robot_model is not None:
            self._robot_model.reset_to_state(state.robot)
        if self._pedestrians_model is not None:
            self._pedestrians_model.reset_to_state(state.pedestrians)
        self._robot_to_world_collision = state.robot_to_world_collision
        self._current_state = self._get_simulation_state({})

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
        self._current_state = self._get_simulation_state(self._current_state.sensors)

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
        self._current_state = self._get_simulation_state(self._current_state.sensors)

    def _make_steps(self, control: Optional[np.ndarray]):
        self._world_map.step(self._sim_dt)

        if self._robot_model is not None:
            self._robot_to_world_collision = not self._robot_model.step(self._sim_dt, self._world_map, control)
            robot_pose = self._robot_model.state.pose
            robot_velocity = self._robot_model.state.velocity
        else:
            robot_pose = None
            robot_velocity = None

        if self._pedestrians_model is not None:
            self._pedestrians_model.step(self._sim_dt, robot_pose, robot_velocity)

    def _get_simulation_state(self, previous_sensors_state: Dict) -> SimulationState:
        world_state = self._get_world_state()
        sensors_state = self._update_sensors_state(world_state, self._world_map, previous_sensors_state)
        return SimulationState(world=world_state, sensors=sensors_state)

    def _get_world_state(self) -> WorldState:
        if self._robot_model is not None:
            if self._pedestrians_model is not None:
                collisions = [k for k, v in self._pedestrians_model.state.poses.items()
                              if np.linalg.norm(self._robot_model.state.pose[:2] - v[:2])
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

        return WorldState(world_map=self._world_map.current_state,
                          robot=robot_state,
                          pedestrians=pedestrians_state,
                          robot_to_pedestrians_collisions=collisions,
                          robot_to_world_collision=self._robot_to_world_collision)

    def _update_sensors_state(self, world_state: WorldState, world_map: AbstractWorldMap,
                              previous_sensors_state: Dict[str, SensorState]) -> Dict[str, SensorState]:
        if self._robot_model is None:
            return {}
        if len(previous_sensors_state) == 0:
            return {sensor.sensor_name: SensorState(reading=sensor.get_reading(world_state, world_map), hold_time=0.)
                    for sensor in self._sensors}

        new_state = {}
        for sensor in self._sensors:
            if sensor.sensor_name not in previous_sensors_state:
                new_state[sensor.sensor_name] = SensorState(reading=sensor.get_reading(world_state, world_map),
                                                            hold_time=0.)
            else:
                sensor_state = previous_sensors_state[sensor.sensor_name]
                hold_time = sensor_state.hold_time + self._sim_dt
                if hold_time >= sensor.period:
                    reading = sensor.get_reading(world_state, world_map)
                    hold_time = 0.
                else:
                    reading = sensor_state.reading
                new_state[sensor.sensor_name] = SensorState(reading, hold_time)

        return new_state
