import numpy as np
from dynamics import agent_dynamics, target_dynamics, integrate_step
from neural_network import NeuralNetwork

class Entity:
    def __init__(self, initial_position, initial_velocity, time_steps, config):
        self.num_states = config['num_states']
        self.positions = np.zeros((self.num_states, time_steps))
        self.velocities = np.zeros((self.num_states, time_steps))
        self.accelerations = np.zeros((self.num_states, time_steps))
        self.positions[:, 0] = initial_position
        self.velocities[:, 0] = initial_velocity
        self.time_step_delta = config['time_step_delta']

    def integrate_dynamics(self, step, acceleration):
        self.accelerations[:, step] = acceleration
        y0 = np.concatenate((self.positions[:, step - 1], self.velocities[:, step - 1]))

        def dynamics(t, y):
            n = len(y) // 2
            return np.concatenate((y[n:], acceleration))
        
        y_final = integrate_step(y0, step, self.time_step_delta, dynamics)
        n = len(y_final) // 2
        self.positions[:, step] = y_final[:n]
        self.velocities[:, step] = y_final[n:]

class Agent(Entity):
    def __init__(self, initial_position, initial_velocity, time_steps, config, target, agent_type):
        super().__init__(initial_position, initial_velocity, time_steps, config)
        self.target = target
        self.agent_type = agent_type
        self.control_size = config['control_size']
        self.k1, self.k2, self.k3 = (config['k1'], config['k2'], config['forgetting_factor'])
        zero_control_output = np.zeros(self.control_size)
        _, self.g, self.g_plus = agent_dynamics(initial_position, initial_velocity, zero_control_output, 0, self.time_step_delta, self.num_states, self.control_size)        
        self.control_output = np.zeros(self.num_states)
        self.tracking_error = np.zeros(self.num_states)

        neural_network_input = lambda step: np.concatenate([self.positions[:, step -1], self.velocities[:, step -1], self.target.positions[:, step -1], self.target.velocities[:, step -1]]).reshape(-1, 1)
        self.neural_network = NeuralNetwork(neural_network_input, config)
        self.neural_network_output = np.zeros(self.num_states)

    def compute_control_output(self, step):
        # Retrieve Gains
        k1, k2, k3 = self.k1, self.k2, self.k3

        # Compute errors.
        self.tracking_error = self.target.positions[:, step - 1] - self.positions[:, step - 1]
        tracking_error_derivative = self.target.velocities[:, step - 1] - self.velocities[:, step - 1]
        filtered_tracking_error = tracking_error_derivative + k1*self.tracking_error

        # Compute neural network output
        loss = filtered_tracking_error
        regularization = - k3 * self.neural_network.weights
        self.neural_network_output = self.neural_network.compute_neural_network_output(step, loss.reshape(-1, 1), regularization).reshape(-1)

        # Compute the controller.
        self.control_output = self.g_plus @ ((1 - k1**2)*self.tracking_error + (k1 + k2)*filtered_tracking_error + self.neural_network_output)

    def update_dynamics(self, step):
        acceleration, self.g, self.g_plus = agent_dynamics(self.positions[:, step - 1], self.velocities[:, step - 1], self.control_output, step, self.time_step_delta, self.num_states, self.control_size)
        self.integrate_dynamics(step, acceleration)

class Target(Entity):
    def __init__(self, initial_position, initial_velocity, time_steps, config):
        super().__init__(initial_position, initial_velocity, time_steps, config)

    def update_dynamics(self, step):
        acceleration = target_dynamics(self.positions[:, step - 1], self.velocities[:, step - 1], self.num_states)
        self.integrate_dynamics(step, acceleration)