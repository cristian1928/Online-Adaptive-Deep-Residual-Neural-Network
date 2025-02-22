import numpy as np
from dynamics import agent_dynamics, target_dynamics, integrate_step
from neural_network import NeuralNetwork

class Entity:
    def __init__(self, num_states, initial_position, time_steps, time_step_delta):
        self.positions = np.zeros((num_states, time_steps))
        self.velocities = np.zeros((num_states, time_steps))
        self.positions[:, 0] = initial_position
        self.time_step_delta = time_step_delta
        
    def integrate_dynamics(self, step, velocity):
        self.velocities[:, step] = velocity
        y0 = self.positions[:, step - 1]
        def dynamics(t, y): return velocity
        y_final = integrate_step(y0, step, self.time_step_delta, dynamics)
        self.positions[:, step] = y_final

class Agent(Entity):
    def __init__(self, num_states, initial_position, time_steps, target, config):
        super().__init__(num_states, initial_position, time_steps,config['time_step_delta'])
        self.num_states = num_states
        self.target = target
        self.k1, self.forgetting_factor = (config['k1'], config['forgetting_factor'])
        self.control_output = np.zeros(num_states)
        self.tracking_error = np.zeros(num_states)
        neural_network_input = lambda step: np.concatenate([self.tracking_error]).reshape(-1, 1)
        self.neural_network = NeuralNetwork(neural_network_input, config)
        self.neural_network_output = np.zeros(config['num_outputs'])

    def compute_control_output(self, step):
        # Compute tracking error.
        self.tracking_error = self.target.positions[:, step - 1] - self.positions[:, step - 1]

        # Compute the neural network output.        
        loss = self.tracking_error
        regularization = - self.forgetting_factor * self.neural_network.weights
        self.neural_network_output = self.neural_network.compute_neural_network_output(step, loss.reshape(-1, 1), regularization).reshape(-1)

        # Compute the controller.
        self.control_output = self.k1*self.tracking_error + self.neural_network_output

    def update_dynamics(self, step):
        velocity = agent_dynamics(self.control_output)
        self.integrate_dynamics(step, velocity)

class Target(Entity):
    def __init__(self, num_states, initial_position, time_steps, time_step_delta):
        super().__init__(num_states, initial_position, time_steps, time_step_delta)
        self.dynamics = np.zeros(num_states)

    def update_dynamics(self, step):
        self.dynamics = target_dynamics(step, self.time_step_delta)
        self.integrate_dynamics(step, self.dynamics)