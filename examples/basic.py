import sys

sys.path.append('..')

import time
from typing import Tuple

import numpy as np

from pyminisim.core import Simulation
from pyminisim.world_map import EmptyWorld, CirclesWorld
from pyminisim.robot import UnicycleRobotModel
from pyminisim.pedestrians import HeadedSocialForceModelPolicy, RandomWaypointTracker, FixedWaypointTracker
from pyminisim.sensors import PedestrianDetectorNoise, PedestrianDetector, \
    LidarSensor, LidarSensorNoise
from pyminisim.visual import Renderer, CircleDrawing


def create_sim() -> Tuple[Simulation, Renderer]:
    robot_model = UnicycleRobotModel(initial_pose=np.array([0., 0., 0.0]),
                                     initial_control=np.array([0.0, np.deg2rad(25.0)]))

    tracker = RandomWaypointTracker(world_size=(7.0, 7.0))
    n_pedestrians = 2
    waypoints = np.zeros((n_pedestrians, 2, 2))
    waypoints[0, :, :] = np.array([[3., 3.],
                                   [-3., -3.]])
    waypoints[1, :, :] = np.array([[-3., -3.],
                                   [3., 3.]])
    pedestrians_model = HeadedSocialForceModelPolicy(n_pedestrians=2,
                                                     waypoint_tracker=tracker,
                                                     pedestrian_linear_velocity_magnitude=np.array([1.5, 2.5]),
                                                     initial_poses=np.array([[-3., -3., 0.],
                                                                             [3., 3., 0.]]))
    # You can model sensor's noise
    # pedestrian_detector_noise = PedestrianDetectorNoise(distance_mu=0., distance_sigma=0.2,
    #                                                     angle_mu=0., angle_sigma=0.05,
    #                                                     misdetection_prob=0.1)
    pedestrian_detector_noise = None
    sensors = [PedestrianDetector(noise=pedestrian_detector_noise)]  # LidarSensor(noise=LidarSensorNoise())]
    sim = Simulation(world_map=EmptyWorld(),  # CirclesWorld(circles=np.array([[2., 2., 1.]])),
                     robot_model=robot_model,
                     pedestrians_model=pedestrians_model,
                     sensors=sensors,
                     rt_factor=1.)
    renderer = Renderer(simulation=sim,
                        resolution=80.0,
                        screen_size=(500, 500))
    return sim, renderer


def main():
    sim, renderer = create_sim()
    renderer.initialize()

    running = True
    sim.step()  # First step can take some time due to Numba compilation
    start_time = time.time()
    end_time = time.time()
    n_frames = 0
    while running:
        renderer.render()
        n_frames += 1
        sim.step()
        current_time = time.time()
        if current_time - start_time >= 1000.0:
            end_time = current_time
            break
    print("FPS: ", n_frames / (end_time - start_time))

    # Done! Time to quit.
    renderer.close()


if __name__ == '__main__':
    main()
