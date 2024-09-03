import sys

sys.path.append('..')

import time
from typing import Tuple

import numpy as np
import pygame

from pyminisim.core import Simulation
from pyminisim.world_map import EmptyWorld, CirclesWorld
from pyminisim.robot import UnicycleRobotModel
from pyminisim.pedestrians import HeadedSocialForceModelPolicy, RandomWaypointTracker, FixedWaypointTracker, ORCAPedestriansModel, ORCAParams
from pyminisim.sensors import PedestrianDetectorNoise, PedestrianDetector, \
    LidarSensor, LidarSensorNoise
from pyminisim.visual import Renderer, CircleDrawing

DEFAULT_CONFIG_PATH = r"configs/mpc_config.yaml"
DEFAULT_RESULT_PATH = r"results/rvo.gif"
DEFAULT_COLOR_HEX_PALETTE = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']


def create_sim() -> Tuple[Simulation, Renderer]:
    robot_model = UnicycleRobotModel(initial_pose=np.array([10., 10., 0.0]),
                                     initial_control=np.array([0.0, np.deg2rad(25.0)]))

    pedestrians_goals= [[[-4,  0]],                              # Provide goals for the pedestrians according to the amount specified in total_peds, [x, y], [m, m]
                        [[ 0,  4]],
                        [[ 0, -4]],
                        [[-2.82, -2.82]],
                        [[ 2.82,  2.82]],
                        [[-2.82,  2.82]],
                        [[ 2.82, -2.82]],
                    ]

    #tracker = RandomWaypointTracker(world_size=(7.0, 7.0))
    n_pedestrians = 7
    waypoints = np.array(pedestrians_goals)
    # Provide initial states of the pedestrians according to the amount specified in total_peds, [x, y, phi], [m, m, rad]
    initial_poses = np.array([(4, 0, 0),
                              (0, -4, 0),
                              (0, 4, 0),
                              (2.82, 2.82, 0),
                              (-2.82, -2.82, 0),
                              (2.82, -2.82, 0),
                              (-2.82, 2.82, 0)])
    tracker = FixedWaypointTracker(initial_positions=initial_poses[:, :2],
                                   waypoints=waypoints,
                                   loop=True)
    pedestrians_model = ORCAPedestriansModel(0.01,
                                                            tracker,
                                                            n_pedestrians,
                                                            initial_poses=initial_poses,
                                                            params=ORCAParams(default_max_speed=2.),
                                                            max_speeds=np.random.uniform(1., 1.8, size=(n_pedestrians)))
    """
    pedestrians_model = HeadedSocialForceModelPolicy(n_pedestrians=2,
                                                     waypoint_tracker=tracker,
                                                     initial_poses=np.array([[-3., -3., 0.],
                                                                             [3., 3., 0.]]))
    """
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
                        screen_size=(700, 700))
    return sim, renderer

def main():
    x_peds = []
    y_peds = []
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
        if current_time - start_time >= 40.0:
            end_time = current_time
            break
        if n_frames % 5 == 0:
            x_peds.append([sim._pedestrians_model._rvo_sim.getAgentPosition(i)[0] for i in range(sim._pedestrians_model._rvo_sim.getNumAgents())])
            y_peds.append([sim._pedestrians_model._rvo_sim.getAgentPosition(i)[1] for i in range(sim._pedestrians_model._rvo_sim.getNumAgents())])
        events = pygame.event.get()
        for event in events:
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    pygame.quit()
    print("FPS: ", n_frames / (end_time - start_time))

    # Done! Time to quit.
    renderer.close()


    import matplotlib as mpl
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation

    # Customizing Matplotlib:
    mpl.rcParams['font.size'] = 18
    mpl.rcParams['lines.linewidth'] = 3
    mpl.rcParams['axes.grid'] = True

    # set figure
    fig, ax = plt.subplots(figsize=[16, 16], facecolor='white')
    ax.set_aspect('equal', adjustable='box')
    fig.suptitle("RVO", fontsize=35)

    # animation function
    cnt = 0

    annotation_offset: np.ndarray = np.array([0, 0.2])

    def plot_pedestrian(x_ped_plt, y_ped_plt, cnt, i) -> None:
        # plot pedestrian i position
        ax.plot(x_ped_plt[:cnt], y_ped_plt[:cnt], linewidth=3,
                color=DEFAULT_COLOR_HEX_PALETTE[i], label=f'Pedestrian {i+1}')
        # plot pedestrian i area
        ped1_radius_plot = plt.Circle(
            (x_ped_plt[cnt], y_ped_plt[cnt]), 0.3, fill=False, linewidth=5, color=DEFAULT_COLOR_HEX_PALETTE[i])
        ax.add_patch(ped1_radius_plot)
        # annotate pedestrian i
        ped_coord = (round(x_ped_plt[cnt], 2), (round(y_ped_plt[cnt], 2)))
        ax.annotate(f'Pedestrian {i+1}: {ped_coord}', ped_coord +
                    np.array([0,  0.3]) + annotation_offset,  ha='center')

    def animate(i) -> None:
        nonlocal cnt
        ax.clear()
        ax.set_xlim([-5, 5])
        ax.set_ylim([-5, 5])

        for ind in range(sim._pedestrians_model._rvo_sim.getNumAgents()):
            plot_pedestrian(np.array(x_peds)[:, ind], np.array(y_peds)[:, ind], cnt, ind)

        # legend
        ax.set_xlabel('$y$ [m]')
        ax.set_ylabel('$x$ [m]')
        ax.legend()
        # increment counter
        cnt = cnt + 1

    print("make_animation: Start")
    anim = FuncAnimation(fig, animate, frames=len(x_peds)-2, interval=0.01, repeat=False)
    anim.save(DEFAULT_RESULT_PATH, 'pillow', len(x_peds)-2)
    print("make_animation: Done")


if __name__ == '__main__':
    main()
