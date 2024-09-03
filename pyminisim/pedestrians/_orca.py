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
    neighbor_dist: float = 1.
    max_neighbors: int = 10
    time_horizon: float = 2.
    time_horizon_obst: float = 2.
    radius = PEDESTRIAN_RADIUS
    default_max_speed: float = 1.7
    goal_reaching_threshold: float = 0.2



# @dataclass
# class ORCAParams:
#     neighbor_dist: float
#     max_neighbors: int
#     time_horizon: float
#     time_horizon_obst: float
#     radius: float
#     max_speed: float
#     velocity: tuple
#     safe_distance: float
#     min_reach_tolerance: float

#     @staticmethod
#     def create_default():
#         return ORCAParams(neighbor_dist=10.,
#                           max_neighbors=10,
#                           time_horizon=5.,
#                           time_horizon_obst=5.,
#                           radius=PEDESTRIAN_RADIUS,
#                           max_speed=30.,
#                           velocity=(0., 0.),
#                           safe_distance=0.,
#                           min_reach_tolerance=0.2)


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


# class OptimalReciprocalCollisionAvoidance(AbstractPedestriansModel):
#     def __init__(self,
#                  dt: float,
#                  waypoint_tracker: AbstractWaypointTracker,
#                  n_pedestrians: int,
#                  initial_poses: Optional[np.ndarray] = None,
#                  initial_velocities: Optional[np.ndarray] = None,
#                  orca_params: ORCAParams = ORCAParams.create_default(),
#                  robot_visible: bool = True) -> None:
#         super().__init__()

#         if initial_poses is None:
#             random_positions = waypoint_tracker.sample_independent_points(n_pedestrians, 0.5)
#             random_orientations = np.random.uniform(-np.pi, np.pi, size=n_pedestrians)
#             initial_poses = np.hstack([random_positions, random_orientations.reshape(-1, 1)])
#         else:
#             assert initial_poses.shape[0] == n_pedestrians
#         if initial_velocities is None:
#             initial_velocities = np.zeros((n_pedestrians, 3))
#         else:
#             assert initial_velocities.shape[0] == n_pedestrians

#         self._params = orca_params
#         self._n_pedestrians = initial_poses.shape[0]
#         self._waypoint_tracker = waypoint_tracker
#         if self._waypoint_tracker.state is None:
#             self._waypoint_tracker.resample_all({i: initial_poses[i] for i in range(self._n_pedestrians)})
#         self._robot_visible = robot_visible
#         self._memory_direction = np.zeros([n_pedestrians, 1])

#         self._rvo_sim = rvo2.PyRVOSimulator(timeStep=dt,
#                                             neighborDist=self._params.neighbor_dist,
#                                             maxNeighbors=self._params.max_neighbors,
#                                             timeHorizon=self._params.time_horizon,
#                                             timeHorizonObst=self._params.time_horizon_obst,
#                                             radius=self._params.radius + self._params.safe_distance,
#                                             maxSpeed=self._params.max_speed,
#                                             velocity=self._params.velocity)

#         # Add pedestrians
#         for i in range(self._n_pedestrians):
#             self._rvo_sim.addAgent(pos=tuple(initial_poses[i]),
#                                    neighborDist=self._params.neighbor_dist,
#                                    maxNeighbors=self._params.max_neighbors,
#                                    timeHorizon=self._params.time_horizon,
#                                    timeHorizonObst=self._params.time_horizon_obst,
#                                    radius=self._params.radius + self._params.safe_distance,
#                                    maxSpeed=np.random.uniform(3., 8.),
#                                    velocity=initial_velocities[i])
#         # if robot_visible:
#         #     self._rvo_sim.addAgent(pos=(0, 0),
#         #                            neighborDist=None,
#         #                            maxNeighbors=None,
#         #                            timeHorizon=None,
#         #                            timeHorizonObst=None,
#         #                            radius=ROBOT_RADIUS,
#         #                            maxSpeed=None,
#         #                            velocity=(0, 0))

#         self.set_preferred_velocities()
#         self._state = ORCAState({i: (initial_poses[i], initial_velocities[i])
#                                  for i in range(self._n_pedestrians)}, self._waypoint_tracker.state)

#     @property
#     def state(self) -> ORCAState:
#         return self._state

#     def set_preferred_velocities(self) -> np.ndarray:

#         vel_array = np.zeros([self._n_pedestrians, 2])
#         # Set the preferred velocity for each agent.
#         for i in range(self._n_pedestrians):
#             vector = np.array(self._waypoint_tracker.state.current_waypoints[i]) - np.array(
#                 self._rvo_sim.getAgentPosition(i))
#             if np.linalg.norm(vector) < self._params.min_reach_tolerance:
#                 # Agent is within one radius of its goal, set preferred velocity to zero
#                 vel = (0, 0)
#                 self._rvo_sim.setAgentPrefVelocity(i, vel)
#             else:
#                 vel = tuple((vector / np.linalg.norm(vector)) * np.random.uniform(5., 10.))
#                 self._rvo_sim.setAgentPrefVelocity(i, vel)
#             vel_array[i, :] = vel
#         return vel_array

#     def step(self,
#              dt: float = None,  # I left this parameter to save the implementation
#              robot_pose: Optional[np.ndarray] = None,
#              robot_velocity: Optional[np.ndarray] = None):

#         self._rvo_sim.doStep()
#         velocities = self.set_preferred_velocities()
#         velocities = np.hstack([velocities, np.zeros([self._n_pedestrians, 1])])

#         poses = np.array([self._rvo_sim.getAgentPosition(agent_no) for agent_no in range(self._n_pedestrians)])
#         poses = np.hstack([poses, np.zeros([self._n_pedestrians, 1])])

#         # Assign pedestrian direction
#         for ped in range(self._n_pedestrians):
#             # If pedestrian reached the goal than save last direction
#             if velocities[ped, 1] == 0 and velocities[ped, 0] == 0:
#                 poses[ped, 2] = self._memory_direction[ped]
#             # Else assign direction of the velocity vector
#             else:
#                 poses[ped, 2] = np.arctan2(velocities[ped, 1], velocities[ped, 0])

#         self._memory_direction = poses[:, 2]

#         if self._waypoint_tracker.state is not None:
#             self._waypoint_tracker.update_waypoints({i: poses[i, :] for i in range(self._n_pedestrians)})

#         self._state = ORCAState({i: (poses[i], velocities[i])
#                                  for i in range(self._n_pedestrians)}, self._waypoint_tracker.state)

#         if self._robot_visible:
#             # As robot is added as the last agent, it is index would be equal to the number of total pedestrians
#             self._rvo_sim.setAgentPosition(self._n_pedestrians, tuple(robot_pose))
#             self._rvo_sim.setAgentPrefVelocity(self._n_pedestrians, tuple(robot_velocity))

#     def reset_to_state(self, state: ORCAState):
#         self._state = state
#         self._waypoint_tracker.reset_to_state(state.waypoints)
