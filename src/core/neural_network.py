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
        self.initialize_weights()
        self.neural_network_gradient_wrt_weights: (NDArray[np.float64] | None) = None
        max_lr = config['maximum_learning_rate']
        min_lr = config['minimum_learning_rate']
        self.beta_1: float = (max_lr * min_lr**3) / (max_lr**2 - min_lr**2)
        self.beta_2: float = min_lr
        self.beta_3: float = (min_lr * max_lr) / (max_lr**2 - min_lr**2)

        initial_lr = config['initial_learning_rate']
        eye_matrix = np.eye(np.size(self.weights))
        self.learning_rate: NDArray[np.float64] = ((initial_lr * eye_matrix)[None, :, :].repeat(self.time_steps, axis=0))
        
        # Pre-allocate temporary arrays to avoid dynamic allocation during forward/backward passes
        self._preallocate_temporary_arrays()

    def initialize_weights(self) -> None:
        activation_to_variance: dict[str, int] = {'tanh': 1, 'sigmoid': 1, 'identity': 1, 'swish': 2, 'relu': 2, 'leaky_relu': 2}
        inner_variance = activation_to_variance[self.inner_layer_activation_function]
        output_variance = activation_to_variance[self.outer_layer_activation_function]
        
        # Pre-calculate total weight size to allocate once
        total_weight_size = 0
        for block in range(self.num_blocks + 1):
            input_size = self.num_inputs if block == 0 else self.num_outputs
            # First layer
            total_weight_size += (input_size + 1) * self.num_neurons
            # Hidden layers
            total_weight_size += (self.num_layers - 1) * (self.num_neurons + 1) * self.num_neurons
            # Output layer
            total_weight_size += (self.num_neurons + 1) * self.num_outputs
        
        # Allocate all weights at once
        all_weights = np.zeros((total_weight_size, 1))
        current_idx = 0
        
        for block in range(self.num_blocks + 1):
            input_size = self.num_inputs if block == 0 else self.num_outputs
            
            # First layer
            layer_size = (input_size + 1) * self.num_neurons
            all_weights[current_idx:current_idx + layer_size, 0] = self.generate_initialized_weights(input_size, self.num_neurons, inner_variance).flatten()
            current_idx += layer_size
            
            # Hidden layers
            for _ in range(self.num_layers - 1):
                layer_size = (self.num_neurons + 1) * self.num_neurons
                all_weights[current_idx:current_idx + layer_size, 0] = self.generate_initialized_weights(self.num_neurons, self.num_neurons, inner_variance).flatten()
                current_idx += layer_size
            
            # Output layer
            layer_size = (self.num_neurons + 1) * self.num_outputs
            all_weights[current_idx:current_idx + layer_size, 0] = self.generate_initialized_weights(self.num_neurons, self.num_outputs, output_variance).flatten()
            current_idx += layer_size
        
        self.weights: NDArray[np.float64] = all_weights

    def _preallocate_temporary_arrays(self) -> None:
        """Pre-allocate all temporary arrays used in forward/backward propagation to avoid dynamic allocation."""
        # Pre-allocate arrays for forward propagation
        max_layer_size = max(self.num_inputs + 1, self.num_neurons + 1, self.num_outputs)
        self._temp_activated_layers: list[list[NDArray[np.float64]]] = []
        self._temp_unactivated_layers: list[list[NDArray[np.float64]]] = []
        
        # Pre-allocate for each layer in each block
        for _ in range(self.num_blocks + 1):
            # Each block has num_layers + 1 layers (including output)
            block_activated: list[NDArray[np.float64]] = []
            block_unactivated: list[NDArray[np.float64]] = []
            
            # Input layer (largest possible size)
            block_activated.append(np.zeros((max_layer_size, 1)))
            
            # Hidden layers + output layer
            for layer_idx in range(self.num_layers + 1):
                if layer_idx < self.num_layers:
                    # Hidden layer
                    block_activated.append(np.zeros((self.num_neurons + 1, 1)))  # +1 for bias
                    block_unactivated.append(np.zeros((self.num_neurons, 1)))
                else:
                    # Output layer
                    block_unactivated.append(np.zeros((self.num_outputs, 1)))
            
            self._temp_activated_layers.append(block_activated)
            self._temp_unactivated_layers.append(block_unactivated)
        
        # Pre-allocate weight matrices (largest possible size for reuse)
        max_rows = max(self.num_inputs + 1, self.num_neurons + 1, self.num_outputs + 1)
        max_cols = max(self.num_neurons, self.num_outputs)
        self._temp_weight_matrices: list[NDArray[np.float64]] = []
        for _ in range(self.num_layers + 1):
            self._temp_weight_matrices.append(np.zeros((max_cols, max_rows)))
        
        # Pre-allocate identity matrices for kronecker products (most common optimization)
        self._temp_kron_eye_neurons: NDArray[np.float64] = np.eye(self.num_neurons)
        self._temp_kron_eye_outputs: NDArray[np.float64] = np.eye(self.num_outputs, dtype=np.float64)
        
        # Pre-allocate neural network output to avoid repeated allocation in forward pass
        self._temp_neural_network_output: NDArray[np.float64] = np.zeros((self.num_outputs, 1))

    def generate_initialized_weights(self, input_size: int, output_size: int, variance_factor: int) -> NDArray[np.float64]:
        # Applies either Xavier (1/input) or He (2/input) initialization
        variance = variance_factor / input_size  
        # input_size + 1 accounts for bias term
        return np.random.normal(0, np.sqrt(variance), output_size * (input_size + 1)).reshape(-1, 1)

    def get_input_with_bias(self, step: int) -> NDArray[np.float64]: 
        return np.append(self.input_func(step), 1).reshape(-1, 1)

    def construct_transposed_weight_matrices(self, weight_index: int) -> tuple[int, list[NDArray[np.float64]]]:
        weight_matrices: list[NDArray[np.float64]] = []
        biased_input_size = self.num_inputs + 1 if weight_index == 0 else self.num_outputs + 1
        biased_neuron_size = self.num_neurons + 1
        layer_shapes = [(biased_input_size, self.num_neurons)] + [(biased_neuron_size, self.num_neurons)] * (self.num_layers - 1) + [(biased_neuron_size, self.num_outputs)]
        
        for layer_idx, (rows, cols) in enumerate(layer_shapes):
            # Extract weights and reshape using pre-allocated array slice
            weight_slice = self.weights[weight_index:weight_index + rows * cols]
            
            # Reuse pre-allocated matrix by taking appropriate slice and reshaping
            matrix_view = self._temp_weight_matrices[layer_idx][:cols, :rows]
            matrix_view[:] = weight_slice.reshape(rows, cols, order='F').T
            
            # Create view of the correctly sized portion
            weight_matrices.append(matrix_view[:cols, :rows].copy())
            weight_index += rows * cols
        return weight_index, weight_matrices

    def perform_forward_propagation(self, transposed_weight_matrices: list[NDArray[np.float64]], input_with_bias: NDArray[np.float64], block_index: int) -> tuple[list[NDArray[np.float64]], list[NDArray[np.float64]]]:
        # Reuse pre-allocated arrays for this block
        activated_layers = self._temp_activated_layers[block_index]
        unactivated_layers = self._temp_unactivated_layers[block_index]
        
        # Reset and set input layer
        input_size = input_with_bias.shape[0]
        activated_layers[0][:input_size, :] = input_with_bias
        
        # Create views for actual computation (to return)
        result_activated = [activated_layers[0][:input_size, :].copy()]
        result_unactivated = []
        
        for layer_index in range(self.num_layers + 1):
            unactivated_output = transposed_weight_matrices[layer_index] @ result_activated[-1]
            
            # Store in pre-allocated array and create result view
            if layer_index < self.num_layers:
                # Hidden layer
                unactivated_layers[layer_index][:self.num_neurons, :] = unactivated_output
                result_unactivated.append(unactivated_layers[layer_index][:self.num_neurons, :].copy())
                
                activation_function = self.outer_layer_activation_function if layer_index == self.num_layers - 1 else self.inner_layer_activation_function
                activated_output = self.apply_activation_function_and_bias(unactivated_output, activation_function)
                
                activated_size = activated_output.shape[0]
                activated_layers[layer_index + 1][:activated_size, :] = activated_output
                result_activated.append(activated_layers[layer_index + 1][:activated_size, :].copy())
            else:
                # Output layer
                unactivated_layers[layer_index][:self.num_outputs, :] = unactivated_output
                result_unactivated.append(unactivated_layers[layer_index][:self.num_outputs, :].copy())
        
        return result_activated, result_unactivated

    def perform_backward_propagation(self, activated_layers: list[NDArray[np.float64]], unactivated_layers: list[NDArray[np.float64]], transposed_weight_matrices: list[NDArray[np.float64]], outer_product: NDArray[np.float64] | None = None) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        gradient: NDArray[np.float64] | None = None
        product: NDArray[np.float64] | None = None
        
        for layer_index in range(self.num_layers, -1, -1):
            transposed_output = activated_layers[layer_index].T
            
            if layer_index == self.num_layers:
                # Compute kronecker product: kron(eye(num_outputs), transposed_output)
                # Result shape: (num_outputs * transposed_output.shape[0], num_outputs * transposed_output.shape[1])
                current_gradient = np.kron(self._temp_kron_eye_outputs, transposed_output)
                
                if outer_product is not None: 
                    gradient = outer_product @ current_gradient
                else: 
                    gradient = current_gradient
                
                product = transposed_weight_matrices[layer_index] @ self.apply_activation_function_derivative_and_bias(unactivated_layers[layer_index - 1], self.outer_layer_activation_function)
            else:
                # Compute kronecker product: kron(eye(num_neurons), transposed_output)
                kron_product = np.kron(self._temp_kron_eye_neurons, transposed_output)
                
                if outer_product is None: 
                    layer_gradient = product @ kron_product if product is not None else np.zeros_like(kron_product)
                else: 
                    layer_gradient = outer_product @ product @ kron_product if product is not None else np.zeros_like(kron_product)
                
                gradient = np.hstack((layer_gradient, gradient)) if gradient is not None else layer_gradient
                
                if layer_index != 0 and product is not None: 
                    product = product @ transposed_weight_matrices[layer_index] @ self.apply_activation_function_derivative_and_bias(unactivated_layers[layer_index - 1], self.inner_layer_activation_function)
        
        if gradient is None: gradient = np.zeros((1, 1))  # fallback case
        if product is None: product = np.zeros((1, 1))  # fallback case
        return gradient, product

    def _run_forward_pass(self, step: int) -> tuple[int, NDArray[np.float64], list[list[NDArray[np.float64]]], list[list[NDArray[np.float64]]], list[list[NDArray[np.float64]]]]:
        weight_index = 0
        # Reuse pre-allocated output array
        self._temp_neural_network_output.fill(0.0)
        neural_network_output = self._temp_neural_network_output
        
        activated_layers_blocks: list[list[NDArray[np.float64]]] = [[] for _ in range(self.num_blocks + 1)]
        unactivated_layers_blocks: list[list[NDArray[np.float64]]] = [[] for _ in range(self.num_blocks + 1)]
        transposed_weights_blocks: list[list[NDArray[np.float64]]] = [[] for _ in range(self.num_blocks + 1)]
        
        for block_index in range(self.num_blocks + 1):
            weight_index, weights_block = self.construct_transposed_weight_matrices(weight_index)
            transposed_weights_blocks[block_index] = weights_block
            input_data = self.get_input_with_bias(step) if block_index == 0 else self.apply_activation_function_and_bias(neural_network_output, self.shortcut_activation_function)
            activated_block, unactivated_block = self.perform_forward_propagation(weights_block, input_data, block_index)
            activated_layers_blocks[block_index] = activated_block
            unactivated_layers_blocks[block_index] = unactivated_block
            neural_network_output += unactivated_block[-1]
        return weight_index, neural_network_output.copy(), activated_layers_blocks, unactivated_layers_blocks, transposed_weights_blocks

    def _run_backward_pass(self, activated_layers_blocks: list[list[NDArray[np.float64]]], unactivated_layers_blocks: list[list[NDArray[np.float64]]], transposed_weights_blocks: list[list[NDArray[np.float64]]]) -> NDArray[np.float64]:
        outer_product: NDArray[np.float64] | None = None
        total_gradient: NDArray[np.float64] | None = None
        for block_index in range(self.num_blocks, -1, -1):
            current_outer_product = outer_product if block_index < self.num_blocks else None
            block_gradient, inner_product = self.perform_backward_propagation(activated_layers_blocks[block_index], unactivated_layers_blocks[block_index], transposed_weights_blocks[block_index], current_outer_product)
            if block_index == self.num_blocks: 
                total_gradient = block_gradient
            else: 
                total_gradient = np.hstack((block_gradient, total_gradient)) if total_gradient is not None else block_gradient
            if block_index > 0:
                block_output = sum(unactivated_layers_blocks[i][-1] for i in range(block_index))
                if isinstance(block_output, (int, float)):
                    # Convert scalar to array for compatibility with activation function
                    block_output = np.array([[block_output]])
                preactivation_derivative = self.apply_activation_function_derivative_and_bias(block_output, self.shortcut_activation_function)
                update_term = inner_product @ transposed_weights_blocks[block_index][0] @ preactivation_derivative
                if block_index == self.num_blocks: 
                    outer_product = np.eye(self.num_outputs) + update_term
                else: 
                    outer_product = outer_product @ (np.eye(self.num_outputs) + update_term) if outer_product is not None else np.eye(self.num_outputs) + update_term
        
        if total_gradient is None:
            total_gradient = np.zeros((1, 1))  # fallback case
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
        _, neural_network_output, activated_layers_blocks, unactivated_layers_blocks, transposed_weights_blocks = self._run_forward_pass(step)
        total_gradient = self._run_backward_pass(activated_layers_blocks, unactivated_layers_blocks, transposed_weights_blocks)
        return total_gradient

    def update_learning_rate(self, step: int) -> None:
        def learning_rate_deriv(t: float, gamma: NDArray[np.float64] | float) -> NDArray[np.float64] | float:
            gamma_arr = np.asarray(gamma)
            if self.neural_network_gradient_wrt_weights is None:
                return np.zeros_like(gamma_arr)
            normalized_neural_network_gradient_wrt_weights = self.neural_network_gradient_wrt_weights / (1.0 + np.linalg.norm(self.neural_network_gradient_wrt_weights.T @ self.neural_network_gradient_wrt_weights, 'fro')**2)
            mat = normalized_neural_network_gradient_wrt_weights @ gamma_arr
            result = - mat.T @ mat + (self.beta_1 * np.eye(gamma_arr.shape[0])) + (self.beta_2 * gamma_arr) - (self.beta_3 * gamma_arr @ gamma_arr)
            return result if isinstance(gamma, np.ndarray) else float(result)
        
        new_lr = integrate_step(self.learning_rate[step - 1], step, self.time_step_delta, learning_rate_deriv)
        self.learning_rate[step] = np.asarray(new_lr)

    def update_neural_network_weights(self, step: int, loss: NDArray[np.float64]) -> None:
        def weights_deriv(t: float, weights: NDArray[np.float64] | float) -> NDArray[np.float64] | float:
            weights_arr = np.asarray(weights)
            if self.neural_network_gradient_wrt_weights is None:
                return np.zeros_like(weights_arr)
            weight_derivative = self.learning_rate[step] @ (self.neural_network_gradient_wrt_weights.T @ loss)
            projected_weights = self.proj(weight_derivative, weights_arr, self.weight_bounds)
            return projected_weights if isinstance(weights, np.ndarray) else float(projected_weights)
        
        new_weights = integrate_step(self.weights, step, self.time_step_delta, weights_deriv)
        self.weights = np.asarray(new_weights)

    def proj(self, Theta: NDArray[np.float64], thetaHat: NDArray[np.float64], thetaBar: float) -> NDArray[np.float64]:
        max_term = max(0.0, (np.dot(thetaHat.T, thetaHat)).item() - thetaBar**2)
        dot_term = (np.dot(thetaHat.T, Theta)).item()
        numerator = max_term**2 * (dot_term + np.sqrt(dot_term**2 + 1.0)) * thetaHat
        denominator = 2.0 * (1.0 + 2.0 * thetaBar)**2 * thetaBar**2
        result = Theta - (numerator / denominator)
        return np.asarray(result)

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
        
        # Create diagonal matrix and append zeros for bias - use minimal allocation
        result_flat = result.flatten()
        diag_size = len(result_flat)
        
        # Create the result matrix with diagonal and zeros
        final_result = np.zeros((diag_size + 1, diag_size))
        np.fill_diagonal(final_result[:diag_size, :], result_flat)
        
        return final_result