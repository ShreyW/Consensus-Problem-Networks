# # communication_graphs = [
# #     [[0, 1, 1, 0, 0], [1, 0, 0, 0, 0], [1, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]],  # Graph 1
# #     [[0, 0, 0, 0, 0], [0, 0, 1, 0, 0], [0, 1, 0, 0, 1], [0, 0, 0, 0, 0], [0, 0, 1, 0, 0]],  # Graph 2
# #     [[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 1], [0, 0, 0, 1, 0]],  # Graph 3
# #     [[0, 0, 0, 0, 0], [0, 0, 1, 0, 0], [0, 1, 0, 1, 0], [0, 0, 1, 0, 0], [0, 0, 0, 0, 0]]   # Graph 4
# # ]


import numpy as np
import matplotlib.pyplot as plt

class SecondOrderMinConsensus:
    def __init__(self, num_agents, initial_states, initial_velocities, switching_function):
        self.num_agents = num_agents
        self.states = initial_states
        self.velocities = initial_velocities
        self.switching_function = switching_function
        self.states_history = [initial_states.copy()]
        self.velocities_history = [initial_velocities.copy()]
        self.controls_history = []

    def min_consensus_protocol(self, t):
        communication_graph = self.switching_function(t)
        u = np.zeros(self.num_agents)
        for i in range(self.num_agents):
            neighbors = communication_graph[i]
            neighbor_values = [self.velocities[j] + self.states[j] for j in neighbors + [i]]
            u[i] = -2 * self.velocities[i] - self.states[i] + min(neighbor_values) # Control Equation
        self.controls_history.append(u)
        # print("Time: %0.3f" %t, "\tControls:", np.round(u,3), "\tGraph Agents Involved", communication_graph)
        return u

# Example switching function
def switching_function(t):
    T0 = 12  # Time period for each graph
    s = int(t // T0)
    t_prime = t - s * T0
    if 0 <= t_prime < T0 / 4:
        return {
            0: [1, 2],
            1: [0],
            2: [0],
            3: [],
            4: []
        }
    elif T0 / 4 <= t_prime < 2 * T0 / 4:
        return {
            0: [],
            1: [2],
            2: [1,4],
            3: [],
            4: [2]
        }
    elif 2 * T0 / 4 <= t_prime < 3 * T0 / 4:
        return {
            0: [],
            1: [],
            2: [],
            3: [4],
            4: [3]
        }
    else:
        return {
            0: [],
            1: [2],
            2: [1,3],
            3: [2],
            4: []
        }

# Example usage:
num_agents = 5
initial_states = np.array([10.0, 7.5, 2.5, -2.5, -7.5])
initial_velocities = np.zeros(num_agents)
# initial_velocities = np.array([1.0, 2.0, 3.5, -7.5, 0]) # Starting with non-zero initial velocities

min_consensus = SecondOrderMinConsensus(num_agents, initial_states, initial_velocities, switching_function)

# Simulation time
timesteps = 60000
dt = 0.001

# Simulation loop
for t in range(timesteps):
    u = min_consensus.min_consensus_protocol(t * dt)
    # Update states using numerical integration
    min_consensus.velocities += u * dt
    min_consensus.states += min_consensus.velocities * dt
    # Save states and velocities for plotting
    min_consensus.states_history.append(min_consensus.states.copy())
    min_consensus.velocities_history.append(min_consensus.velocities.copy())

# Plotting
t_values = np.arange(0, timesteps * dt + dt, dt)
states_history = np.array(min_consensus.states_history)
velocities_history = np.array(min_consensus.velocities_history)
controls_history = np.array(min_consensus.controls_history)

# Create subplots
fig, axs = plt.subplots(3, figsize=(12, 12))

# Plot states
for i in range(num_agents):
    axs[0].plot(t_values, states_history[:, i], label=f"Agent {i+1} State")

# Plot velocities
for i in range(num_agents):
    axs[1].plot(t_values, velocities_history[:, i], label=f"Agent {i+1} Velocity")

# Plot controls
for i in range(num_agents):
    axs[2].plot(t_values[:-1], controls_history[:, i], label=f"Agent {i+1} Control", linestyle='--')

# Set titles and labels
axs[0].set_title('States over Time')
axs[1].set_title('Velocities over Time')
axs[2].set_title('Controls over Time')

for ax in axs:
    ax.set_xlabel('Time')
    ax.set_ylabel('Values')
    ax.legend()
    ax.yaxis.grid(True)
    ax.set_xlim(0)

plt.tight_layout()
plt.show()
