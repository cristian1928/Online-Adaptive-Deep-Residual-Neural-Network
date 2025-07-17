<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
from typing import Any, Callable, Dict, cast

=======
>>>>>>> parent of ed2ed2d (Merge pull request #14 from cristian1928/copilot/fix-12)
=======
>>>>>>> parent of ed2ed2d (Merge pull request #14 from cristian1928/copilot/fix-12)
=======
>>>>>>> parent of ed2ed2d (Merge pull request #14 from cristian1928/copilot/fix-12)
import numpy as np
from typing import Dict, Any, Callable
import dynamics
from integrate import integrate_step
from neural_network import NeuralNetwork

class Entity:
    def __init__(self, initial_position: np.ndarray, time_steps: int, config: Dict[str, Any]) -> None:
        self.num_states: int = config['num_states']
        self.time_step_delta: float = config['time_step_delta']
        self.positions: np.ndarray = np.zeros((self.num_states, time_steps))
        self.velocities: np.ndarray = np.zeros((self.num_states, time_steps))
        self.positions[:, 0] = initial_position

class Agent(Entity):
    def __init__(self, initial_position: np.ndarray, time_steps: int, config: Dict[str, Any], target: 'Target', agent_type: str) -> None:
        super().__init__(initial_position, time_steps, config)
        self.target: Target = target
        self.agent_type: str = agent_type
        self.k1: float = config['k1']
        self.control_output: np.ndarray = np.zeros(self.num_states)
        self.tracking_error: np.ndarray = np.zeros(self.num_states)
        self.neural_network: NeuralNetwork = NeuralNetwork(self._input_func, config)
        self.neural_network_output: np.ndarray = np.zeros(self.num_states)

    def _input_func(self, step: int) -> np.ndarray:
        return self.target.positions[:, step - 1]

    def compute_control_output(self, step: int) -> None:
        self.tracking_error = self.target.positions[:, step - 1] - self.positions[:, step - 1]
        loss: np.ndarray = self.tracking_error
        self.neural_network_output = self.neural_network.train_step(step, loss.reshape(-1, 1)).reshape(-1)
        self.control_output = self.k1*self.tracking_error + self.neural_network_output

    def update_dynamics(self, step: int) -> None:
        result = integrate_step(self.positions[:, step - 1], step, self.time_step_delta, 
                               lambda t, y: self.control_output)
        self.positions[:, step] = result

class Target(Entity):
    def __init__(self, initial_position: np.ndarray, time_steps: int, config: Dict[str, Any]) -> None:
        super().__init__(initial_position, time_steps, config)
        dynamics_type: str = config.get('dynamics_type', 'trophic_dynamics')
        self.dynamics_function: Callable[[np.ndarray], np.ndarray] = dynamics.get_dynamics_function(dynamics_type)
        
    def update_dynamics(self, step: int) -> None:
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
        dynamics_func = cast(
            Callable[[float, np.ndarray], np.ndarray], lambda t, pos: self.dynamics_function(pos)
        )
        result = integrate_step(
            self.positions[:, step - 1],
            step,
            self.time_step_delta,
            dynamics_func,
        )
        self.positions[:, step] = result
=======
        result = integrate_step(self.positions[:, step - 1], step, self.time_step_delta, 
                               lambda t, pos: self.dynamics_function(pos))
        self.positions[:, step] = result
>>>>>>> parent of ed2ed2d (Merge pull request #14 from cristian1928/copilot/fix-12)
=======
        result = integrate_step(self.positions[:, step - 1], step, self.time_step_delta, 
                               lambda t, pos: self.dynamics_function(pos))
        self.positions[:, step] = result
>>>>>>> parent of ed2ed2d (Merge pull request #14 from cristian1928/copilot/fix-12)
=======
        result = integrate_step(self.positions[:, step - 1], step, self.time_step_delta, 
                               lambda t, pos: self.dynamics_function(pos))
        self.positions[:, step] = result
>>>>>>> parent of ed2ed2d (Merge pull request #14 from cristian1928/copilot/fix-12)
