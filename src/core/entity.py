import numpy as np
import numpy.typing as npt
from typing import Dict, Any, Callable
from ..simulation import dynamics
from ..simulation.integrate import integrate_step
from .neural_network import NeuralNetwork

class Entity:
    def __init__(self, initial_position: npt.NDArray[np.floating[Any]], time_steps: int, config: Dict[str, Any]) -> None:
        self.num_states: int = config['num_states']
        self.time_step_delta: float = config['time_step_delta']
        self.positions: npt.NDArray[np.floating[Any]] = np.zeros((self.num_states, time_steps))
        self.velocities: npt.NDArray[np.floating[Any]] = np.zeros((self.num_states, time_steps))
        self.positions[:, 0] = initial_position

class Agent(Entity):
    def __init__(self, initial_position: npt.NDArray[np.floating[Any]], time_steps: int, config: Dict[str, Any], target: 'Target', agent_type: str) -> None:
        super().__init__(initial_position, time_steps, config)
        self.target: 'Target' = target
        self.agent_type: str = agent_type
        self.k1: float = config['k1']
        self.control_output: npt.NDArray[np.floating[Any]] = np.zeros(self.num_states)

        self.tracking_error: npt.NDArray[np.floating[Any]] = np.zeros(self.num_states)

        self.neural_network: NeuralNetwork = NeuralNetwork(self._input_func, config)
        self.neural_network_output: npt.NDArray[np.floating[Any]] = np.zeros(self.num_states)

    def _input_func(self, step: int) -> npt.NDArray[np.floating[Any]]: 
        return self.target.positions[:, step - 1]

    def compute_control_output(self, step: int) -> None:
        # Compute tracking error
        self.tracking_error = self.target.positions[:, step - 1] - self.positions[:, step - 1]

        # Neural network update
        loss = self.tracking_error
        self.neural_network_output = self.neural_network.train_step(step, loss.reshape(-1, 1)).reshape(-1)

        # Compute Controller
        self.control_output = self.k1*self.tracking_error + self.neural_network_output

    def update_dynamics(self, step: int) -> None: 
        self.positions[:, step] = integrate_step(self.positions[:, step - 1], step, self.time_step_delta, lambda t, y: self.control_output)

class Target(Entity):
    def __init__(self, initial_position: npt.NDArray[np.floating[Any]], time_steps: int, config: Dict[str, Any]) -> None:
        super().__init__(initial_position, time_steps, config)
        dynamics_type = config.get('dynamics_type', 'trophic_dynamics')
        self.dynamics_function: Callable[[npt.NDArray[np.floating[Any]]], npt.NDArray[np.floating[Any]]] = dynamics.get_dynamics_function(dynamics_type)
        
    def update_dynamics(self, step: int) -> None: 
        self.positions[:, step] = integrate_step(self.positions[:, step - 1], step, self.time_step_delta, lambda t, pos: self.dynamics_function(pos))