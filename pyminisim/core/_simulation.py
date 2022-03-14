from typing import Optional, List, Dict, Tuple
import time
from dataclasses import dataclass

import numpy as np

# from pyminisim.core import AbstractRobotMotionModel, AbstractPedestriansPolicy, AbstractWaypointTracker, \
#     AbstractSensorReading, AbstractSensor, ROBOT_RADIUS, PEDESTRIAN_RADIUS
from ._world_state import WorldState
from ._simulation_state import SimulationState
from ._motion import AbstractRobotMotionModel
from ._pedestrians_policy import AbstractPedestriansPolicy
from ._waypoints import AbstractWaypointTracker
from ._sensor import AbstractSensorReading, AbstractSensor
from ._constants import ROBOT_RADIUS, PEDESTRIAN_RADIUS


class Simulation:

    def __init__(self,
                 robot_model: Optional[AbstractRobotMotionModel],
                 pedestrians_model: Optional[AbstractPedestriansPolicy],
                 sensors: List[AbstractSensor],
                 sim_dt: float = 0.01,
                 rt_factor: Optional[float] = 1.0):
        self._robot_model = robot_model
        self._pedestrians_model = pedestrians_model
        self._sensors = sensors
        self._sim_dt = sim_dt
        self._rt_factor = rt_factor

        self._current_state = self._get_simulation_state()

    @property
    def current_state(self) -> SimulationState:
        return self._current_state

    @property
    def sensors(self) -> List[AbstractSensor]:
        return self._sensors

    def set_control(self, control: np.ndarray):
        if self._robot_model is not None:
            self._robot_model.control = control

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
        sensors_readings = self._get_sensors_readings(world_state)
        return SimulationState(world=world_state, sensors=sensors_readings)

    def _get_world_state(self) -> WorldState:
        if self._robot_model is not None:
            if self._pedestrians_model is not None:
                collisions = [i for i, e in enumerate(self._pedestrians_model.poses)
                              if np.linalg.norm(self._robot_model.state.pose[:2] - e[:2])
                              < (ROBOT_RADIUS + PEDESTRIAN_RADIUS)]
            else:
                collisions = None
            robot_pose = self._robot_model.state.pose
            robot_velocity = self._robot_model.state.velocity
            last_control = self._robot_model.state.control
        else:
            collisions = None
            robot_pose = None
            robot_velocity = None
            last_control = None

        if self._pedestrians_model is not None:
            pedestrians_poses = self._pedestrians_model.poses
            pedestrians_velocities = self._pedestrians_model.velocities
        else:
            pedestrians_poses = None
            pedestrians_velocities = None

        return WorldState(robot_pose=robot_pose,
                          robot_velocity=robot_velocity,
                          pedestrians_poses=pedestrians_poses,
                          pedestrians_velocities=pedestrians_velocities,
                          last_control=last_control,
                          robot_to_pedestrians_collisions=collisions)

    def _get_sensors_readings(self, world_state: WorldState) -> Dict:
        if self._robot_model is not None:
            return {sensor.sensor_name: sensor.get_reading(world_state) for sensor in self._sensors}
        else:
            return {}
