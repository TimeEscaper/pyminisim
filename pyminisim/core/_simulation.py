from typing import Optional, List, Dict, Tuple
import time
from dataclasses import dataclass

import numpy as np

from pyminisim.core import AbstractRobotMotionModel, AbstractPedestriansPolicy, AbstractWaypointTracker, \
    AbstractSensor, ROBOT_RADIUS, PEDESTRIAN_RADIUS


@dataclass
class SimulationState:
    robot_pose: np.ndarray
    robot_velocity: np.ndarray
    pedestrians_poses: np.ndarray
    pedestrians_velocities: np.ndarray
    last_control: np.ndarray
    robot_to_pedestrians_collisions: List[int]


class Simulation:

    def __init__(self,
                 robot_model: AbstractRobotMotionModel,
                 pedestrians_model: AbstractPedestriansPolicy,
                 waypoint_tracker: AbstractWaypointTracker,
                 sensors: List[AbstractSensor],
                 sim_dt: float = 0.01,
                 rt_factor: Optional[float] = 1.0):
        self._robot_model = robot_model
        self._pedestrians_model = pedestrians_model
        self._waypoint_tracker = waypoint_tracker
        self._sensors = sensors
        self._sim_dt = sim_dt
        self._rt_factor = rt_factor

        self._waypoint_tracker.update_waypoints(self._pedestrians_model.poses)

    def set_control(self, control: np.ndarray):
        self._robot_model.control = control

    def step(self) -> Tuple[SimulationState, Dict]:
        time_start = time.time()
        self._make_steps()
        time_end = time.time()
        time_elapsed = time_end - time_start
        if self._rt_factor is not None:
            time_to_sleep = self._sim_dt / self._rt_factor - time_elapsed
            if time_to_sleep > 0.:
                # TODO: Log think on logging RT factor info
                time.sleep(self._sim_dt / self._rt_factor)

        simulation_state = self._get_simulation_state()
        readings = self._get_sensors_readings(simulation_state)

        return simulation_state, readings

    def _make_steps(self):
        self._robot_model.step(self._sim_dt)
        self._pedestrians_model.step(self._sim_dt, self._robot_model.pose, self._robot_model.velocity)
        self._waypoint_tracker.update_waypoints(self._pedestrians_model.poses)

    def _get_simulation_state(self) -> SimulationState:
        collisions = [i for i, e in enumerate(self._pedestrians_model.poses)
                      if np.linalg.norm(self._robot_model.pose[:2] - e[:2]) < (ROBOT_RADIUS + PEDESTRIAN_RADIUS)]
        return SimulationState(robot_pose=self._robot_model.pose,
                               robot_velocity=self._robot_model.velocity,
                               pedestrians_poses=self._pedestrians_model.poses,
                               pedestrians_velocities=self._pedestrians_model.velocities,
                               last_control=self._robot_model.control,
                               robot_to_pedestrians_collisions=collisions)

    def _get_sensors_readings(self, simulation_state: SimulationState) -> Dict:
        return {sensor.sensor_name: sensor.get_reading(simulation_state) for sensor in self._sensors}
