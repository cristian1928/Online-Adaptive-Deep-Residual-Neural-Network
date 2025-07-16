import numpy as np
import dynamics
from integrate import integrate_step
from neural_network import NeuralNetwork

class Entity:
    def __init__(self, initial_position, time_steps, config):
        self.num_states = config['num_states']
        self.time_step_delta = config['time_step_delta']
        self.positions = np.zeros((self.num_states, time_steps))
        self.velocities = np.zeros((self.num_states, time_steps))
        self.positions[:, 0] = initial_position

class Agent(Entity):
    def __init__(self, initial_position, time_steps, config, target, agent_type):
        super().__init__(initial_position, time_steps, config)
        self.target = target
        self.agent_type = agent_type
        self.k1 = config['k1']
        self.control_output = np.zeros(self.num_states)

        self.tracking_error = np.zeros(self.num_states)

        self.neural_network = NeuralNetwork(self._input_func, config)
        self.neural_network_output = np.zeros(self.num_states)

    def _input_func(self, step): return self.target.positions[:, step - 1]

    def compute_control_output(self, step):
        # Compute tracking error
        self.tracking_error = self.target.positions[:, step - 1] - self.positions[:, step - 1]

        # Neural network update
        loss = self.tracking_error
        self.neural_network_output = self.neural_network.train_step(step, loss.reshape(-1, 1)).reshape(-1)

        # Compute Controller
        self.control_output = self.k1*self.tracking_error + self.neural_network_output

    def update_dynamics(self, step): self.positions[:, step] = integrate_step(self.positions[:, step - 1], step, self.time_step_delta, lambda t, y: self.control_output)

class Target(Entity):
    def update_dynamics(self, step): self.positions[:, step] = integrate_step(self.positions[:, step - 1], step, self.time_step_delta, lambda t, pos: dynamics.trophic_dynamics(pos))