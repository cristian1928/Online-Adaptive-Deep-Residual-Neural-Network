from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
from numpy.typing import NDArray

from ..simulation import dynamics
from ..simulation.integrate import integrate_step
from .neural_network import NeuralNetwork


class Entity:
    def __init__(self, initial_position: NDArray[np.float64], time_steps: int, config: dict[str, Any]) -> None:
        self.num_states: int = config['num_states']
        self.time_step_delta: float = config['time_step_delta']
        self.positions: NDArray[np.float64] = np.zeros((self.num_states, time_steps))
        self.velocities: NDArray[np.float64] = np.zeros((self.num_states, time_steps))
        self.positions[:, 0] = initial_position

class Agent(Entity):
    def __init__(self, initial_position: NDArray[np.float64], time_steps: int, config: dict[str, Any], target: "Target", agent_type: str) -> None:
        super().__init__(initial_position, time_steps, config)
        self.target: "Target" = target
        self.agent_type: str = agent_type
        self.k1: float = config['k1']
        self.control_output: NDArray[np.float64] = np.zeros(self.num_states)
        self.tracking_error: NDArray[np.float64] = np.zeros(self.num_states)
        self.neural_network: NeuralNetwork = NeuralNetwork(self._input_func, config)
        self.neural_network_output: NDArray[np.float64] = np.zeros(self.num_states)

    def _input_func(self, step: int) -> NDArray[np.float64]: return self.target.positions[:, step - 1]

    def compute_control_output(self, step: int) -> None:
        self.tracking_error = (self.target.positions[:, step - 1] - self.positions[:, step - 1])
        self.control_output = self.k1*self.tracking_error

        if self.agent_type == "Proportional": return

        loss = self.tracking_error
        nn_output = self.neural_network.train_step(step, loss.reshape(-1, 1))
        self.neural_network_output = nn_output.reshape(-1)
        self.control_output += self.neural_network_output

    def update_dynamics(self, step: int) -> None: 
        def control_wrapper(t: float, y: NDArray[np.float64]) -> NDArray[np.float64]:
            return self.control_output
        self.velocities[:, step] = self.control_output
        result = integrate_step(self.positions[:, step - 1], step, self.time_step_delta, control_wrapper)
        self.positions[:, step] = result

class Target(Entity):
    def __init__(self, initial_position: NDArray[np.float64], time_steps: int, config: dict[str, Any]) -> None:
        super().__init__(initial_position, time_steps, config)
        dynamics_type = config['dynamics_type']
        self.dynamics_function: Callable[[NDArray[np.float64]], NDArray[np.float64]] = dynamics.get_dynamics_function(dynamics_type)
        
    def update_dynamics(self, step: int) -> None: 
        def dynamics_wrapper(t: float, pos: NDArray[np.float64]) -> NDArray[np.float64]:
            return self.dynamics_function(pos)
        self.velocities[:, step] = self.dynamics_function(self.positions[:, step - 1])
        result = integrate_step(self.positions[:, step - 1], step, self.time_step_delta, dynamics_wrapper)
        self.positions[:, step] = result