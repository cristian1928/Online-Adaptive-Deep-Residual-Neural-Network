from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
from numpy.typing import NDArray

from ..simulation.integrate import integrate_step


class NeuralNetwork:
    def __init__(self, input_func: Callable[[int], NDArray[np.float64]], config: dict[str, Any]) -> None:
        self.time_step_delta: float = config['time_step_delta']
        self.time_steps: int = int(config['final_time'] / self.time_step_delta)
        self.input_func: Callable[[int], NDArray[np.float64]] = input_func
        self.inner_layer_activation_function: str = config['inner_activation']
        self.outer_layer_activation_function: str = config['output_activation']
        self.shortcut_activation_function: str = config['shortcut_activation']
        self.num_blocks: int = config['num_blocks']
        self.num_layers: int = config['num_layers']
        self.num_neurons: int = config['num_neurons']
        self.num_inputs: int = input_func(1).shape[0]
        self.num_outputs: int = config['output_size']
        self.weight_bounds: float = config['weight_bounds']
        
        # Create a deterministic random number generator for this neural network
        # Use the configuration's ID and global seed to ensure deterministic but unique initialization
        base_seed = config.get('seed', 0)
        agent_id = config.get('ID', 'default')
        # Create a hash-based seed from the agent ID to ensure different agents get different but deterministic seeds
        import hashlib
        id_hash = int(hashlib.md5(agent_id.encode()).hexdigest()[:8], 16)
        self.rng = np.random.RandomState(base_seed + id_hash)
        
        self.initialize_weights()
        self.neural_network_gradient_wrt_weights: NDArray[np.float64] = np.zeros((self.num_outputs, np.size(self.weights)))
        mu_min = config['minimum_singular_value']
        mu_max = config['maximum_singular_value']
        self.alpha: float = (mu_max * mu_min**3) / (mu_max**2 - mu_min**2)
        self.beta: float = mu_min
        self.gamma: float = (mu_min * mu_max) / (mu_max**2 - mu_min**2)

        initial_lr_matrix = config['initial_learning_rate'] * np.eye(np.size(self.weights))
        self.learning_rate = np.stack([initial_lr_matrix] * self.time_steps, axis=0)

    def initialize_weights(self) -> None:
        activation_to_variance: dict[str, int] = {'tanh': 1, 'sigmoid': 1, 'identity': 1, 'swish': 2, 'relu': 2, 'leaky_relu': 2}
        inner_variance = activation_to_variance[self.inner_layer_activation_function]
        output_variance = activation_to_variance[self.outer_layer_activation_function]
        weights: list[NDArray[np.float64]] = []
        for block in range(self.num_blocks + 1):
            input_size = self.num_inputs if block == 0 else self.num_outputs
            weights.append(self.generate_initialized_weights(input_size, self.num_neurons, inner_variance))
            for _ in range(self.num_layers - 1):
                weights.append(self.generate_initialized_weights(self.num_neurons, self.num_neurons, inner_variance))
            weights.append(
                self.generate_initialized_weights(self.num_neurons, self.num_outputs, output_variance))
        self.weights: NDArray[np.float64] = np.vstack(weights)

    def generate_initialized_weights(self, input_size: int, output_size: int, variance_factor: int) -> NDArray[np.float64]:
        variance = variance_factor / input_size  # Applies either Xavier (1/input) or He (2/input) initialization
        return self.rng.normal(0, np.sqrt(variance), output_size * (input_size + 1)).reshape(-1, 1)    # input_size + 1 accounts for bias term

    def get_input_with_bias(self, step: int) -> NDArray[np.float64]: 
        return np.append(self.input_func(step), 1).reshape(-1, 1)

    def construct_transposed_weight_matrices(self, weight_index: int) -> tuple[int, list[NDArray[np.float64]]]:
        weight_matrices: list[NDArray[np.float64]] = []
        biased_input_size = self.num_inputs + 1 if weight_index == 0 else self.num_outputs + 1
        biased_neuron_size = self.num_neurons + 1
        layer_shapes = [(biased_input_size, self.num_neurons)] + [(biased_neuron_size, self.num_neurons)] * (self.num_layers - 1) + [(biased_neuron_size, self.num_outputs)]
        for rows, cols in layer_shapes:
            matrix = np.array(self.weights[weight_index:weight_index + rows * cols]).reshape(rows, cols, order='F')
            weight_matrices.append(matrix.T)
            weight_index += rows * cols
        return weight_index, weight_matrices

    def perform_forward_propagation(self, transposed_weight_matrices: list[NDArray[np.float64]], input_with_bias: NDArray[np.float64]) -> tuple[list[NDArray[np.float64]], list[NDArray[np.float64]]]:
        activated_layers: list[NDArray[np.float64]] = [input_with_bias]
        unactivated_layers: list[NDArray[np.float64]] = []
        for layer_index in range(self.num_layers + 1):
            unactivated_output = transposed_weight_matrices[layer_index] @ activated_layers[-1]
            unactivated_layers.append(unactivated_output)
            if layer_index != self.num_layers:
                activation_function = self.outer_layer_activation_function if layer_index == self.num_layers - 1 else self.inner_layer_activation_function
                activated_layers.append(self.apply_activation_function_and_bias(unactivated_output, activation_function))
        return activated_layers, unactivated_layers

    def perform_backward_propagation(self, activated_layers: list[NDArray[np.float64]], unactivated_layers: list[NDArray[np.float64]], transposed_weight_matrices: list[NDArray[np.float64]], outer_product: NDArray[np.float64]) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        transposed_output_last_layer = activated_layers[self.num_layers].T
        gradient_base = np.kron(np.eye(self.num_outputs), transposed_output_last_layer)
        last_layer_gradient = outer_product @ gradient_base
        layer_gradients: list[NDArray[np.float64]] = [last_layer_gradient]
        product = (transposed_weight_matrices[self.num_layers] @ self.apply_activation_function_derivative_and_bias(unactivated_layers[self.num_layers - 1], self.outer_layer_activation_function))
        for layer_index in range(self.num_layers - 1, -1, -1):
            transposed_output = activated_layers[layer_index].T
            kron_product = np.kron(np.eye(self.num_neurons), transposed_output)
            hidden_layer_gradient = outer_product @ product @ kron_product
            layer_gradients.append(hidden_layer_gradient)
            if layer_index > 0:
                product = (product @ transposed_weight_matrices[layer_index] @ self.apply_activation_function_derivative_and_bias(unactivated_layers[layer_index - 1], self.inner_layer_activation_function))
        gradient = np.hstack(list(reversed(layer_gradients)))
        return gradient, product

    def _run_forward_pass(self, step: int) -> tuple[int, NDArray[np.float64], list[list[NDArray[np.float64]]], list[list[NDArray[np.float64]]], list[list[NDArray[np.float64]]]]:
        weight_index = 0
        neural_network_output: NDArray[np.float64] = np.zeros(self.num_outputs).reshape(-1, 1)
        activated_layers_blocks: list[list[NDArray[np.float64]]] = [[] for _ in range(self.num_blocks + 1)]
        unactivated_layers_blocks: list[list[NDArray[np.float64]]] = [[] for _ in range(self.num_blocks + 1)]
        transposed_weights_blocks: list[list[NDArray[np.float64]]] = [[] for _ in range(self.num_blocks + 1)]
        
        for block_index in range(self.num_blocks + 1):
            weight_index, weights_block = self.construct_transposed_weight_matrices(weight_index)
            transposed_weights_blocks[block_index] = weights_block
            input_data = self.get_input_with_bias(step) if block_index == 0 else self.apply_activation_function_and_bias(neural_network_output, self.shortcut_activation_function)
            activated_block, unactivated_block = self.perform_forward_propagation(weights_block, input_data)
            activated_layers_blocks[block_index] = activated_block
            unactivated_layers_blocks[block_index] = unactivated_block
            neural_network_output += unactivated_block[-1]
        return weight_index, neural_network_output, activated_layers_blocks, unactivated_layers_blocks, transposed_weights_blocks

    def _run_backward_pass(self, activated_layers_blocks: list[list[NDArray[np.float64]]], unactivated_layers_blocks: list[list[NDArray[np.float64]]], transposed_weights_blocks: list[list[NDArray[np.float64]]]) -> NDArray[np.float64]:
        outer_product: NDArray[np.float64] = np.eye(self.num_outputs)
        gradient_blocks: list[NDArray[np.float64]] = []
        for block_index in range(self.num_blocks, -1, -1):
            block_gradient, inner_product = self.perform_backward_propagation(activated_layers_blocks[block_index], unactivated_layers_blocks[block_index], transposed_weights_blocks[block_index], outer_product)
            gradient_blocks.append(block_gradient)

            if block_index > 0:
                block_output = sum((unactivated_layers_blocks[i][-1] for i in range(block_index)), start=np.array(0.0))
                preactivation_derivative = self.apply_activation_function_derivative_and_bias(block_output, self.shortcut_activation_function)
                update_term = inner_product @ transposed_weights_blocks[block_index][0] @ preactivation_derivative
                outer_product = outer_product @ (np.eye(self.num_outputs) + update_term)
        
        total_gradient = np.hstack(list(reversed(gradient_blocks)))
        return total_gradient

    def predict(self, step: int) -> NDArray[np.float64]:
        self.learning_rate[step] = self.learning_rate[step - 1]
        _, neural_network_output, _, _, _ = self._run_forward_pass(step)
        return neural_network_output

    def train_step(self, step: int, loss: NDArray[np.float64]) -> NDArray[np.float64]:
        _, neural_network_output, activated_layers_blocks, unactivated_layers_blocks, transposed_weights_blocks = self._run_forward_pass(step)
        total_gradient = self._run_backward_pass(activated_layers_blocks, unactivated_layers_blocks, transposed_weights_blocks)
        self.neural_network_gradient_wrt_weights = total_gradient
        self.update_neural_network_weights(step, loss)
        self.update_learning_rate(step)
        return neural_network_output

    def set_weights(self, weights: NDArray[np.float64]) -> None:
        self.weights = weights.copy()

    def forward_raw(self, step: int) -> NDArray[np.float64]:
        _, neural_network_output, _, _, _ = self._run_forward_pass(step)
        return neural_network_output

    def jacobian_raw(self, step: int) -> NDArray[np.float64]:
        _, _, activated_layers_blocks, unactivated_layers_blocks, transposed_weights_blocks = self._run_forward_pass(step)
        total_gradient = self._run_backward_pass(activated_layers_blocks, unactivated_layers_blocks, transposed_weights_blocks)
        return total_gradient

    def update_learning_rate(self, step: int) -> None:
    
        def learning_rate_dynamics(t: float, learning_rate: NDArray[np.float64]) -> NDArray[np.float64]:
            normalized_regressor = self.neural_network_gradient_wrt_weights / np.linalg.norm(self.neural_network_gradient_wrt_weights, 2)
            product = normalized_regressor @ learning_rate
            least_square_term = product.T @ product
            forgetting_term = self.alpha * np.size(self.weights) + self.beta*learning_rate - self.gamma* learning_rate @ learning_rate
            result = -least_square_term + forgetting_term
            return result
    
        new_lr = integrate_step(self.learning_rate[step - 1], step, self.time_step_delta, learning_rate_dynamics)
        self.learning_rate[step] = new_lr

    def update_neural_network_weights(self, step: int, loss: NDArray[np.float64]) -> None:
        def weights_deriv(t: float, weights: NDArray[np.float64]) -> NDArray[np.float64]:
            weight_derivative = self.learning_rate[step] @ (self.neural_network_gradient_wrt_weights.T @ loss)
            projected_weights = self.proj(weight_derivative, weights, self.weight_bounds)
            return projected_weights
        
        new_weights = integrate_step(self.weights, step, self.time_step_delta, weights_deriv)
        self.weights = new_weights

    def proj(self, Theta: NDArray[np.float64], thetaHat: NDArray[np.float64], thetaBar: float) -> NDArray[np.float64]:
        dot_term: np.float64 = np.dot(thetaHat.T, Theta)
        max_term_val: np.float64 = np.dot(thetaHat.T, thetaHat) - thetaBar**2
        max_term: np.float64 = np.maximum(0.0, max_term_val)
        sqrt_argument: np.float64 = dot_term**2 + 1.0
        sqrt_result: np.float64 = np.sqrt(sqrt_argument)
        scalar_multiplier: np.float64 = max_term**2 * (dot_term + sqrt_result)
        numerator = scalar_multiplier * thetaHat
        denominator = 2.0 * (1.0 + 2.0 * thetaBar)**2 * thetaBar**2
        result = Theta - (numerator / denominator)
        return result

    @staticmethod
    def apply_activation_function_and_bias(x: NDArray[np.float64], activation_function: str) -> NDArray[np.float64]:
        if activation_function == 'tanh': 
            result = np.tanh(x)
        elif activation_function == 'swish': 
            result = x * (1.0 / (1.0 + np.exp(-x)))
        elif activation_function == 'identity': 
            result = x
        elif activation_function == 'relu': 
            result = np.maximum(0, x)
        elif activation_function == 'sigmoid': 
            result = 1 / (1 + np.exp(-x))
        elif activation_function == 'leaky_relu': 
            result = np.where(x > 0, x, 0.01 * x)
        else:
            raise ValueError(f"Unknown activation function: {activation_function}")
        return np.vstack((result, [[1]]))

    @staticmethod
    def apply_activation_function_derivative_and_bias(x: NDArray[np.float64], activation_function: str) -> NDArray[np.float64]:
        if activation_function == 'tanh': 
            result = 1 - np.tanh(x)**2
        elif activation_function == 'swish':
            sigmoid = 1.0 / (1.0 + np.exp(-x))
            swish = x * sigmoid
            result = swish + sigmoid * (1 - swish)
        elif activation_function == 'identity': 
            result = np.ones_like(x)
        elif activation_function == 'relu': 
            result = (x > 0).astype(float)
        elif activation_function == 'sigmoid':
            sigmoid = 1 / (1 + np.exp(-x))
            result = sigmoid * (1 - sigmoid)
        elif activation_function == 'leaky_relu': 
            result = np.where(x > 0, 1, 0.01)
        else:
            raise ValueError(f"Unknown activation function: {activation_function}")
        diag_result = np.diag(result.flatten())
        zeros_shape = (1, diag_result.shape[1]) if diag_result.shape[1] > 0 else (1, 1)
        zeros_array = np.zeros(zeros_shape)
        return np.vstack((diag_result, zeros_array))