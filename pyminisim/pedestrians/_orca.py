"""
Optimal Reciprocal Collision Avoidance
Source: https://gamma.cs.unc.edu/ORCA/
"""
import rvo2
from typing import Optional
import numpy as np
from pyminisim.core import AbstractPedestriansModelState, AbstractPedestriansModel, AbstractWaypointTracker
from pyminisim.core import ROBOT_RADIUS, PEDESTRIAN_RADIUS
from dataclasses import dataclass

@dataclass
class ORCAParams:
    neighbor_dist:  float
    max_neighbors: int
    time_horizon: float
    time_horizon_obst: float
    radius: float
    max_speed: float
    velocity: tuple
    safe_distance: float

    @staticmethod
    def create_default():
        return ORCAParams(neighbor_dist = 1.5,
                          max_neighbors = 5,
                          time_horizon = 1.5,
                          time_horizon_obst = 2,
                          radius = PEDESTRIAN_RADIUS,
                          max_speed = 2,
                          velocity = (0, 0))

class ORCAState(AbstractPedestriansModelState):
    pass

class OptimalReciprocalCollisionAvoidance(AbstractPedestriansModel):
    def __init__(self,
                 waypoint_tracker: AbstractWaypointTracker,
                 n_pedestrians: int,
                 initial_poses: Optional[np.ndarray] = None,
                 initial_velocities: Optional[np.ndarray] = None,
                 orca_params: ORCAParams = ORCAParams.create_default(),
                 robot_visible: bool = True) -> None:
        super().__init__()

        if initial_poses is None:
            random_positions = waypoint_tracker.sample_independent_points(n_pedestrians, 0.5)
            random_orientations = np.random.uniform(-np.pi, np.pi, size=n_pedestrians)
            initial_poses = np.hstack([random_positions, random_orientations.reshape(-1, 1)])
        else:
            assert initial_poses.shape[0] == n_pedestrians
        if initial_velocities is None:
            initial_velocities = np.zeros((n_pedestrians, 3))
        else:
            assert initial_velocities.shape[0] == n_pedestrians

        self._params = orca_params
        self._n_pedestrians = initial_poses.shape[0]
        self._waypoint_tracker = waypoint_tracker
        if self._waypoint_tracker.state is None:
            self._waypoint_tracker.resample_all(initial_poses)
        self._robot_visible = robot_visible
        self._rvo_sim = None
    
    @property
    def state(self) -> ORCAState:
        return self._state

    def step(self, 
             dt: float, 
             robot_pose: Optional[np.ndarray], 
             robot_velocity: Optional[np.ndarray]):

        if robot_pose is None:
            assert robot_velocity is None
        elif robot_velocity is None:
            assert robot_pose is None

        if self._rvo_sim is not None and self._rvo_sim.getNumAgents() != len(self.state): #+ 1:
            del self._rvo_sim
            self._rvo_sim = None

        if self._rvo_sim is None:
            self._rvo_sim = rvo2.PyRVOSimulator(dt, 
                                                self._params.neighbor_dist, 
                                                self._params.max_neighbors,
                                                self._params.time_horizon,
                                                self._params.time_horizon_obst, 
                                                self._params.radius, 
                                                self._params.max_speed,
                                                self._params.velocity)
            # Add Robot
            self._rvo_sim.addAgent(pos = robot_pose, 
                                   neighborDist = ROBOT_RADIUS +  self._params.safe_distance,
                                   radius = ROBOT_RADIUS,
                                   velocity = robot_velocity)
       
            # Add pedestrians
            for i in self._n_pedestrians:
                self._rvo_sim.addAgent(pos = self._initial_poses[i],
                                       neighborDist = PEDESTRIAN_RADIUS + self._params.safe_distance,
                                       velocity = self._initial_velocities[i])
        else:
            