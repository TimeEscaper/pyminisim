import rvo2
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Initialize the simulator
sim = rvo2.PyRVOSimulator(0.01, 10., 5, 10., 10., 0.1, 2)

# Add agents
a0 = sim.addAgent((0, 0))
a1 = sim.addAgent((1, 0))
a2 = sim.addAgent((1, 1))
a3 = sim.addAgent((0, 1), 1.5, 5, 1.5, 2, 0.1, 2, (0, 0))

# Set preferred velocities for the agents
sim.setAgentPrefVelocity(a0, (1, 1))
sim.setAgentPrefVelocity(a1, (-1, 1))
sim.setAgentPrefVelocity(a2, (-1, -1))
sim.setAgentPrefVelocity(a3, (1, -1))

# Initialize the plot
fig, ax = plt.subplots()
ax.set_xlim(-2, 2)
ax.set_ylim(-2, 2)

# Assign different colors to the agents
colors = ['b', 'g', 'r', 'c']  # List of colors for different agents

# Plot the agents
agent_plots = [ax.plot([], [], color + 'o', markersize=8)[0] for color in colors]  # colored dots for agents

# Function to initialize the plot
def init():
    for plot in agent_plots:
        plot.set_data([], [])
    return agent_plots

# Function to update the plot
def update(frame):
    sim.doStep()

    # Get agents' positions
    positions = [sim.getAgentPosition(agent_no) for agent_no in (a0, a1, a2, a3)]
    for pos, plot in zip(positions, agent_plots):
        plot.set_data(*pos)

    if frame > 50:
        sim.setAgentPrefVelocity(a0, (-1, -1))

    return agent_plots

# Create the animation and save it as a video file
ani = animation.FuncAnimation(fig, update, frames=range(1000), init_func=init, blit=True, repeat=False)

# Save the animation
ani.save('simulation.mp4', writer='ffmpeg', fps=30)

print('Simulation has %i agents.' % sim.getNumAgents())
print('Running simulation')