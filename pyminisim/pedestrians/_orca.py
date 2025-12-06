"""
Optimal Reciprocal Collision Avoidance
Source: https://gamma.cs.unc.edu/ORCA/
"""
import rvo2
from typing import Optional
import numpy as np
from pyminisim.core import AbstractPedestriansModelState, AbstractPedestriansModel, AbstractWaypointTracker
from pyminisim.core import PEDESTRIAN_RADIUS
from dataclasses import dataclass


@dataclass
class ORCAParams:
    neighbor_dist: float = 1.
    max_neighbors: int = 10
    time_horizon: float = 2.
    time_horizon_obst: float = 2.
    radius = PEDESTRIAN_RADIUS
    default_max_speed: float = 1.7
    goal_reaching_threshold: float = 0.2


class ORCAState(AbstractPedestriansModelState):
    pass


class ORCAPedestriansModel(AbstractPedestriansModel):

    def __init__(self,
                 dt: float,
                 waypoint_tracker: AbstractWaypointTracker,
                 n_pedestrians: int,
                 params = ORCAParams(),
                 initial_poses: Optional[np.ndarray] = None,
                 initial_velocities: Optional[np.ndarray] = None,
                 preferred_speeds: Optional[np.ndarray] = None,
                 max_speeds: Optional[np.ndarray] = None,
                 robot_visible: bool = False) -> None:
        super(ORCAPedestriansModel, self).__init__()

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
        if max_speeds is None:
            max_speeds = np.array([params.default_max_speed for _ in range(n_pedestrians)])
        else:
            assert max_speeds.shape == (n_pedestrians,), f"max_speeds must have shape (n_pedestrians,), got shape {max_speeds.shape}"
        initial_speeds = np.linalg.norm(initial_velocities[:, :2], axis=1)
        assert (initial_speeds <= max_speeds).all(), "Initial speeds must be <= all corresponding max_speeds"
        if preferred_speeds is not None:
            assert preferred_speeds.shape == (n_pedestrians,), f"preferred_speeds must have shape (n_pedestrians,), got shape {preferred_speeds.shape}"
            assert (preferred_speeds <= max_speeds).all(), f"All preferred_speeds must be <= corresponding max_speeds"
        else:
             preferred_speeds = max_speeds.copy()

        self._n_pedestrians = initial_poses.shape[0]
        self._waypoint_tracker = waypoint_tracker
        if self._waypoint_tracker.state is None:
            self._waypoint_tracker.resample_all({i: initial_poses[i] for i in range(self._n_pedestrians)})
        self._preferred_speeds = preferred_speeds
        self._goal_reaching_threshold = params.goal_reaching_threshold
        self._robot_visible = robot_visible
        self._max_speeds = max_speeds

        self._rvo_sim = rvo2.PyRVOSimulator(timeStep=dt,
                                            neighborDist=params.neighbor_dist,
                                            maxNeighbors=params.max_neighbors,
                                            timeHorizon=params.time_horizon,
                                            timeHorizonObst=params.time_horizon_obst,
                                            radius=params.radius,
                                            maxSpeed=params.default_max_speed)
        
        for i in range(n_pedestrians):
            self._rvo_sim.addAgent(tuple(initial_poses[i, :2]),
                                   maxSpeed=max_speeds[i],
                                   velocity=tuple(initial_velocities[i, :2]))

        self._update_preferred_velocities()

        self._state = ORCAState({i: (initial_poses[i], initial_velocities[i])
                                 for i in range(self._n_pedestrians)}, self._waypoint_tracker.state)
    
    @property
    def state(self) -> ORCAState:
        return self._state
    
    def step(self,
             dt: float = None,
             robot_pose: Optional[np.ndarray] = None,
             robot_velocity: Optional[np.ndarray] = None):
        self._rvo_sim.doStep()

        velocities = np.array([self._rvo_sim.getAgentVelocity(i) + (0.,) 
                               for i in range(self._n_pedestrians)])
        poses = np.array([self._rvo_sim.getAgentPosition(i) + (np.arctan2(velocities[i, 1], velocities[i, 0]),)
                          for i in range(self._n_pedestrians)])
        
        if self._waypoint_tracker.state is not None:
            self._waypoint_tracker.update_waypoints({i: poses[i, :] for i in range(self._n_pedestrians)})

        self._state = ORCAState({i: (poses[i], velocities[i])
                                 for i in range(self._n_pedestrians)}, self._waypoint_tracker.state)
        
        self._update_preferred_velocities()

    def reset_to_state(self, state: ORCAState):
        self._state = state
        self._waypoint_tracker.reset_to_state(state.waypoints)

    def _update_preferred_velocities(self):
        for i in range(self._n_pedestrians):
            agent_position = np.array(self._rvo_sim.getAgentPosition(i))
            waypoints = np.array(self._waypoint_tracker.state.current_waypoints[i])
            direction = waypoints - agent_position
            if np.linalg.norm(direction) <= self._goal_reaching_threshold:
                self._rvo_sim.setAgentPrefVelocity(i, (0., 0.))
            else:
                velocity = direction / np.linalg.norm(direction) * self._preferred_speeds[i]
                self._rvo_sim.setAgentPrefVelocity(i, tuple(velocity))
