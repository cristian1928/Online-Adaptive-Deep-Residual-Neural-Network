from __future__ import annotations

from collections.abc import Callable
from typing import Any, cast

import jax
import jax.numpy as jnp
import numpy as np  # Keep for compatibility with typing and specific operations
from numpy.typing import NDArray

# Enable 64-bit precision in JAX for compatibility with existing code
jax.config.update("jax_enable_x64", True)  # type: ignore[no-untyped-call]

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
        
        # Pre-allocate temporary arrays to avoid reshaping in compute_control_output
        self._temp_loss_reshaped: NDArray[np.float64] = np.zeros((self.num_states, 1))
        self._temp_nn_output_flat: NDArray[np.float64] = np.zeros(self.num_states)

    def _input_func(self, step: int) -> NDArray[np.float64]: return self.target.positions[:, step - 1]

    def compute_control_output(self, step: int) -> None:
        # Compute tracking error (in-place operation)
        self.tracking_error[:] = (self.target.positions[:, step - 1] - self.positions[:, step - 1])

        # Neural network update - use pre-allocated arrays to avoid reshaping
        self._temp_loss_reshaped[:, 0] = self.tracking_error
        nn_output = self.neural_network.train_step(step, self._temp_loss_reshaped)
        self.neural_network_output[:] = nn_output[:, 0]  # Extract column without reshaping

        # Compute Controller (in-place operation)
        self.control_output[:] = self.k1 * self.tracking_error + self.neural_network_output

    def update_dynamics(self, step: int) -> None: 
        def control_wrapper(t: float, y: NDArray[np.float64] | float) -> NDArray[np.float64] | float:
            return self.control_output
        result = integrate_step(self.positions[:, step - 1], step, self.time_step_delta, control_wrapper)
        self.positions[:, step] = cast(NDArray[np.float64], result)

class Target(Entity):
    def __init__(self, initial_position: NDArray[np.float64], time_steps: int, config: dict[str, Any]) -> None:
        super().__init__(initial_position, time_steps, config)
        dynamics_type = config.get('dynamics_type', 'trophic_dynamics')
        self.dynamics_function: Callable[[NDArray[np.float64]], NDArray[np.float64]] = dynamics.get_dynamics_function(dynamics_type)
        
    def update_dynamics(self, step: int) -> None: 
        def dynamics_wrapper(t: float, pos: NDArray[np.float64] | float) -> NDArray[np.float64] | float:
            pos_array = np.asarray(pos)
            return self.dynamics_function(pos_array)
        result = integrate_step(self.positions[:, step - 1], step, self.time_step_delta, dynamics_wrapper)
        self.positions[:, step] = cast(NDArray[np.float64], result)