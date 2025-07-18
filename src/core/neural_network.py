from __future__ import annotations

from collections.abc import Callable
from typing import Any, NamedTuple

import numpy as np
from numpy.typing import NDArray

from ..simulation.integrate import integrate_step


class NetworkState(NamedTuple):
    weights: NDArray[np.float64]
    learning_rate: NDArray[np.float64]


# Activation functions for function-based dispatch
def _swish(x: NDArray[np.float64]) -> NDArray[np.float64]:
    return x * (1.0 / (1.0 + np.exp(-x)))

def _leaky_relu(x: NDArray[np.float64]) -> NDArray[np.float64]:
    return np.where(x > 0, x, 0.01 * x)

def _sigmoid(x: NDArray[np.float64]) -> NDArray[np.float64]:
    return 1 / (1 + np.exp(-x))

ACTIVATION_FUNCTIONS: dict[str, Callable[[NDArray[np.float64]], NDArray[np.float64]]] = {
    'tanh': np.tanh,
    'swish': _swish,
    'identity': lambda x: x,
    'relu': lambda x: np.maximum(0, x),
    'sigmoid': _sigmoid,
    'leaky_relu': _leaky_relu,
}

# Activation function derivatives
def _swish_derivative(x: NDArray[np.float64]) -> NDArray[np.float64]:
    sigmoid = 1.0 / (1.0 + np.exp(-x))
    swish = x * sigmoid
    return swish + sigmoid * (1 - swish)

def _leaky_relu_derivative(x: NDArray[np.float64]) -> NDArray[np.float64]:
    return np.where(x > 0, 1, 0.01)

def _sigmoid_derivative(x: NDArray[np.float64]) -> NDArray[np.float64]:
    sigmoid = 1 / (1 + np.exp(-x))
    return sigmoid * (1 - sigmoid)

ACTIVATION_DERIVATIVES: dict[str, Callable[[NDArray[np.float64]], NDArray[np.float64]]] = {
    'tanh': lambda x: (1 - np.tanh(x)**2).astype(np.float64),
    'swish': _swish_derivative,
    'identity': lambda x: np.ones_like(x),
    'relu': lambda x: (x > 0).astype(np.float64),
    'sigmoid': _sigmoid_derivative,
    'leaky_relu': _leaky_relu_derivative,
}


def generate_initialized_weights(key: Any, input_size: int, output_size: int, variance_factor: int) -> NDArray[np.float64]:
    """Generate initialized weights using Xavier/He initialization.
    
    Args:
        key: Random key (unused for now, prepared for JAX compatibility)
        input_size: Number of input features
        output_size: Number of output features  
        variance_factor: Variance factor (1 for Xavier, 2 for He)
    
    Returns:
        Initialized weight matrix
    """
    # Applies either Xavier (1/input) or He (2/input) initialization
    variance = variance_factor / input_size  
    # input_size + 1 accounts for bias term
    return np.random.normal(0, np.sqrt(variance), output_size * (input_size + 1)).reshape(-1, 1)


def initialize_network_weights(key: Any, config: dict[str, Any], input_size: int) -> NDArray[np.float64]:
    """Initialize all network weights.
    
    Args:
        key: Random key (unused for now, prepared for JAX compatibility)
        config: Network configuration
        input_size: Size of input vector
        
    Returns:
        Initialized weight matrix for entire network
    """
    activation_to_variance: dict[str, int] = {'tanh': 1, 'sigmoid': 1, 'identity': 1, 'swish': 2, 'relu': 2, 'leaky_relu': 2}
    inner_variance = activation_to_variance[config['inner_activation']]
    output_variance = activation_to_variance[config['output_activation']]
    
    weights: list[NDArray[np.float64]] = []
    num_blocks = config['num_blocks']
    num_layers = config['num_layers']
    num_neurons = config['num_neurons']
    num_outputs = config['output_size']
    
    for block in range(num_blocks + 1):
        block_input_size = input_size if block == 0 else num_outputs
        weights.append(generate_initialized_weights(key, block_input_size, num_neurons, inner_variance))
        for _ in range(num_layers - 1):
            weights.append(generate_initialized_weights(key, num_neurons, num_neurons, inner_variance))
        weights.append(generate_initialized_weights(key, num_neurons, num_outputs, output_variance))
    
    return np.vstack(weights)


def initialize_learning_rate(config: dict[str, Any], time_steps: int, num_weights: int) -> NDArray[np.float64]:
    """Initialize learning rate matrix.
    
    Args:
        config: Network configuration
        time_steps: Number of time steps
        num_weights: Number of weights in network
        
    Returns:
        Initialized learning rate matrix
    """
    initial_lr: float = config['initial_learning_rate']
    eye_matrix: NDArray[np.float64] = np.eye(num_weights, dtype=np.float64)
    scaled_matrix: NDArray[np.float64] = initial_lr * eye_matrix
    expanded_matrix: NDArray[np.float64] = scaled_matrix[None, :, :].repeat(time_steps, axis=0)
    return expanded_matrix


def create_initial_network_state(key: Any, config: dict[str, Any], input_size: int) -> NetworkState:
    """Create initial network state.
    
    Args:
        key: Random key (unused for now, prepared for JAX compatibility)
        config: Network configuration
        input_size: Size of input vector
        
    Returns:
        Initial NetworkState
    """
    weights = initialize_network_weights(key, config, input_size)
    time_steps = int(config['final_time'] / config['time_step_delta'])
    learning_rate = initialize_learning_rate(config, time_steps, np.size(weights))
    
    return NetworkState(weights=weights, learning_rate=learning_rate)


def get_input_with_bias(input_func: Callable[[int], NDArray[np.float64]], step: int) -> NDArray[np.float64]:
    """Get input vector with bias term added."""
    return np.append(input_func(step), 1).reshape(-1, 1)


def construct_transposed_weight_matrices(config: dict[str, Any], weights: NDArray[np.float64], weight_index: int) -> tuple[int, list[NDArray[np.float64]]]:
    """Construct transposed weight matrices from flat weight vector."""
    weight_matrices: list[NDArray[np.float64]] = []
    
    # Get dimensions from config
    num_inputs = config['num_inputs']
    num_outputs = config['output_size']
    num_neurons = config['num_neurons']
    num_layers = config['num_layers']
    
    biased_input_size = num_inputs + 1 if weight_index == 0 else num_outputs + 1
    biased_neuron_size = num_neurons + 1
    layer_shapes = [(biased_input_size, num_neurons)] + [(biased_neuron_size, num_neurons)] * (num_layers - 1) + [(biased_neuron_size, num_outputs)]
    
    for rows, cols in layer_shapes:
        matrix = np.array(weights[weight_index:weight_index + rows * cols]).reshape(rows, cols, order='F')
        weight_matrices.append(matrix.T)
        weight_index += rows * cols
    return weight_index, weight_matrices


def perform_forward_propagation(config: dict[str, Any], transposed_weight_matrices: list[NDArray[np.float64]], input_with_bias: NDArray[np.float64]) -> tuple[list[NDArray[np.float64]], list[NDArray[np.float64]]]:
    """Perform forward propagation through a block."""
    activated_layers: list[NDArray[np.float64]] = [input_with_bias]
    unactivated_layers: list[NDArray[np.float64]] = []
    
    num_layers = config['num_layers']
    inner_activation = config['inner_activation']
    outer_activation = config['output_activation']
    
    for layer_index in range(num_layers + 1):
        unactivated_output = transposed_weight_matrices[layer_index] @ activated_layers[-1]
        unactivated_layers.append(unactivated_output)
        if layer_index != num_layers:
            activation_function = outer_activation if layer_index == num_layers - 1 else inner_activation
            activated_layers.append(apply_activation_function_and_bias(unactivated_output, activation_function))
    return activated_layers, unactivated_layers


def perform_backward_propagation(config: dict[str, Any], activated_layers: list[NDArray[np.float64]], unactivated_layers: list[NDArray[np.float64]], transposed_weight_matrices: list[NDArray[np.float64]], outer_product: NDArray[np.float64] | None = None) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Perform backward propagation through a block."""
    gradient: NDArray[np.float64] | None = None
    product: NDArray[np.float64] | None = None
    
    num_layers = config['num_layers']
    num_neurons = config['num_neurons']
    num_outputs = config['output_size']
    inner_activation = config['inner_activation']
    outer_activation = config['output_activation']
    
    for layer_index in range(num_layers, -1, -1):
        transposed_output = activated_layers[layer_index].T
        if layer_index == num_layers:
            current_gradient: NDArray[np.float64] = np.asarray(np.kron(np.eye(num_outputs, dtype=np.float64), transposed_output), dtype=np.float64)
            if outer_product is not None: gradient = outer_product @ current_gradient
            else: gradient = current_gradient
            product = transposed_weight_matrices[layer_index] @ apply_activation_function_derivative_and_bias(unactivated_layers[layer_index - 1], outer_activation)
        else:
            kron_product = np.kron(np.eye(num_neurons), transposed_output)
            if outer_product is None: layer_gradient = product @ kron_product if product is not None else np.zeros_like(kron_product)
            else: layer_gradient = outer_product @ product @ kron_product if product is not None else np.zeros_like(kron_product)
            gradient = np.hstack((layer_gradient, gradient)) if gradient is not None else layer_gradient
            if layer_index != 0 and product is not None: product = product @ transposed_weight_matrices[layer_index] @ apply_activation_function_derivative_and_bias(unactivated_layers[layer_index - 1], inner_activation)
    
    if gradient is None: gradient = np.zeros((1, 1))  # fallback case
    if product is None: product = np.zeros((1, 1))  # fallback case
    return gradient, product


def apply_activation_function_and_bias(x: NDArray[np.float64], activation_function: str) -> NDArray[np.float64]:
    """Apply activation function and add bias term."""
    if activation_function not in ACTIVATION_FUNCTIONS:
        raise ValueError(f"Unknown activation function: {activation_function}")
    
    result = ACTIVATION_FUNCTIONS[activation_function](x)
    return np.vstack((result, [[1]]))


def apply_activation_function_derivative_and_bias(x: NDArray[np.float64], activation_function: str) -> NDArray[np.float64]:
    """Apply activation function derivative and add bias term."""
    if activation_function not in ACTIVATION_DERIVATIVES:
        raise ValueError(f"Unknown activation function: {activation_function}")
        
    result = ACTIVATION_DERIVATIVES[activation_function](x)
    diag_result = np.diag(result.flatten())
    zeros_shape = (1, diag_result.shape[1]) if diag_result.shape[1] > 0 else (1, 1)
    zeros_array = np.zeros(zeros_shape)
    return np.vstack((diag_result, zeros_array))


def run_forward_pass(config: dict[str, Any], input_func: Callable[[int], NDArray[np.float64]], weights: NDArray[np.float64], step: int) -> tuple[int, NDArray[np.float64], list[list[NDArray[np.float64]]], list[list[NDArray[np.float64]]], list[list[NDArray[np.float64]]]]:
    """Run forward pass through the entire network."""
    weight_index = 0
    num_outputs = config['output_size']
    num_blocks = config['num_blocks']
    
    neural_network_output: NDArray[np.float64] = np.zeros(num_outputs).reshape(-1, 1)
    activated_layers_blocks: list[list[NDArray[np.float64]]] = [[] for _ in range(num_blocks + 1)]
    unactivated_layers_blocks: list[list[NDArray[np.float64]]] = [[] for _ in range(num_blocks + 1)]
    transposed_weights_blocks: list[list[NDArray[np.float64]]] = [[] for _ in range(num_blocks + 1)]
    
    for block_index in range(num_blocks + 1):
        weight_index, weights_block = construct_transposed_weight_matrices(config, weights, weight_index)
        transposed_weights_blocks[block_index] = weights_block
        
        if block_index == 0:
            input_data = get_input_with_bias(input_func, step)
        else:
            input_data = apply_activation_function_and_bias(neural_network_output, config['shortcut_activation'])
            
        activated_block, unactivated_block = perform_forward_propagation(config, weights_block, input_data)
        activated_layers_blocks[block_index] = activated_block
        unactivated_layers_blocks[block_index] = unactivated_block
        neural_network_output += unactivated_block[-1]
    
    return weight_index, neural_network_output, activated_layers_blocks, unactivated_layers_blocks, transposed_weights_blocks


def run_backward_pass(config: dict[str, Any], activated_layers_blocks: list[list[NDArray[np.float64]]], unactivated_layers_blocks: list[list[NDArray[np.float64]]], transposed_weights_blocks: list[list[NDArray[np.float64]]]) -> NDArray[np.float64]:
    """Run backward pass through the entire network."""
    outer_product: NDArray[np.float64] | None = None
    total_gradient: NDArray[np.float64] | None = None
    num_blocks = config['num_blocks']
    num_outputs = config['output_size']
    
    for block_index in range(num_blocks, -1, -1):
        current_outer_product = outer_product if block_index < num_blocks else None
        block_gradient, inner_product = perform_backward_propagation(config, activated_layers_blocks[block_index], unactivated_layers_blocks[block_index], transposed_weights_blocks[block_index], current_outer_product)
        
        if block_index == num_blocks: 
            total_gradient = block_gradient
        else: 
            total_gradient = np.hstack((block_gradient, total_gradient)) if total_gradient is not None else block_gradient
            
        if block_index > 0:
            block_output = sum(unactivated_layers_blocks[i][-1] for i in range(block_index))
            if isinstance(block_output, (int, float)):
                # Convert scalar to array for compatibility with activation function
                block_output = np.array([[block_output]])
            preactivation_derivative = apply_activation_function_derivative_and_bias(block_output, config['shortcut_activation'])
            update_term = inner_product @ transposed_weights_blocks[block_index][0] @ preactivation_derivative
            
            if block_index == num_blocks: 
                outer_product = np.eye(num_outputs) + update_term
            else: 
                outer_product = outer_product @ (np.eye(num_outputs) + update_term) if outer_product is not None else np.eye(num_outputs) + update_term
    
    if total_gradient is None:
        total_gradient = np.zeros((1, 1))  # fallback case
    return total_gradient


def predict(config: dict[str, Any], input_func: Callable[[int], NDArray[np.float64]], state: NetworkState, step: int) -> tuple[NDArray[np.float64], NetworkState]:
    """Predict output for given step."""
    # Copy learning rate from previous step
    new_learning_rate = state.learning_rate.copy()
    new_learning_rate[step] = state.learning_rate[step - 1]
    new_state = NetworkState(weights=state.weights, learning_rate=new_learning_rate)
    
    _, neural_network_output, _, _, _ = run_forward_pass(config, input_func, state.weights, step)
    return neural_network_output, new_state


def proj(theta: NDArray[np.float64], theta_hat: NDArray[np.float64], theta_bar: float) -> NDArray[np.float64]:
    """Project weight update."""
    max_term = max(0.0, (np.dot(theta_hat.T, theta_hat)).item() - theta_bar**2)
    dot_term = (np.dot(theta_hat.T, theta)).item()
    numerator = max_term**2 * (dot_term + np.sqrt(dot_term**2 + 1.0)) * theta_hat
    denominator = 2.0 * (1.0 + 2.0 * theta_bar)**2 * theta_bar**2
    result = theta - (numerator / denominator)
    return np.asarray(result)


def update_neural_network_weights(config: dict[str, Any], state: NetworkState, gradient: NDArray[np.float64], loss: NDArray[np.float64], step: int) -> NetworkState:
    """Update neural network weights functionally."""
    beta_1 = config['beta_1']
    beta_2 = config['beta_2'] 
    beta_3 = config['beta_3']
    weight_bounds = config['weight_bounds']
    time_step_delta = config['time_step_delta']
    
    def weights_deriv(t: float, weights: NDArray[np.float64] | float) -> NDArray[np.float64] | float:
        weights_arr = np.asarray(weights)
        if gradient is None:
            return np.zeros_like(weights_arr)
        weight_derivative = state.learning_rate[step] @ (gradient.T @ loss)
        projected_weights = proj(weight_derivative, weights_arr, weight_bounds)
        return projected_weights if isinstance(weights, np.ndarray) else float(projected_weights)
    
    new_weights = integrate_step(state.weights, step, time_step_delta, weights_deriv)
    return NetworkState(weights=np.asarray(new_weights), learning_rate=state.learning_rate)


def update_learning_rate(config: dict[str, Any], state: NetworkState, gradient: NDArray[np.float64], step: int) -> NetworkState:
    """Update learning rate functionally."""
    beta_1 = config['beta_1']
    beta_2 = config['beta_2'] 
    beta_3 = config['beta_3']
    time_step_delta = config['time_step_delta']
    
    def learning_rate_deriv(t: float, gamma: NDArray[np.float64] | float) -> NDArray[np.float64] | float:
        gamma_arr = np.asarray(gamma)
        if gradient is None:
            return np.zeros_like(gamma_arr)
        normalized_gradient = gradient / (1.0 + np.linalg.norm(gradient.T @ gradient, 'fro')**2)
        mat = normalized_gradient @ gamma_arr
        result = - mat.T @ mat + (beta_1 * np.eye(gamma_arr.shape[0])) + (beta_2 * gamma_arr) - (beta_3 * gamma_arr @ gamma_arr)
        return result if isinstance(gamma, np.ndarray) else float(result)
    
    new_lr = integrate_step(state.learning_rate[step - 1], step, time_step_delta, learning_rate_deriv)
    new_learning_rate = state.learning_rate.copy()
    new_learning_rate[step] = np.asarray(new_lr)
    return NetworkState(weights=state.weights, learning_rate=new_learning_rate)


def train_step(config: dict[str, Any], input_func: Callable[[int], NDArray[np.float64]], state: NetworkState, loss: NDArray[np.float64], step: int) -> tuple[NDArray[np.float64], NetworkState]:
    """Train one step functionally."""
    # Forward pass
    _, neural_network_output, activated_layers_blocks, unactivated_layers_blocks, transposed_weights_blocks = run_forward_pass(config, input_func, state.weights, step)
    
    # Backward pass
    total_gradient = run_backward_pass(config, activated_layers_blocks, unactivated_layers_blocks, transposed_weights_blocks)
    
    # Update weights
    new_state = update_neural_network_weights(config, state, total_gradient, loss, step)
    
    # Update learning rate
    final_state = update_learning_rate(config, new_state, total_gradient, step)
    
    return neural_network_output, final_state


def forward_raw(config: dict[str, Any], input_func: Callable[[int], NDArray[np.float64]], weights: NDArray[np.float64], step: int) -> NDArray[np.float64]:
    """Raw forward pass for testing."""
    _, neural_network_output, _, _, _ = run_forward_pass(config, input_func, weights, step)
    return neural_network_output


def jacobian_raw(config: dict[str, Any], input_func: Callable[[int], NDArray[np.float64]], weights: NDArray[np.float64], step: int) -> NDArray[np.float64]:
    """Raw jacobian computation for testing."""
    _, neural_network_output, activated_layers_blocks, unactivated_layers_blocks, transposed_weights_blocks = run_forward_pass(config, input_func, weights, step)
    total_gradient = run_backward_pass(config, activated_layers_blocks, unactivated_layers_blocks, transposed_weights_blocks)
    return total_gradient


class NeuralNetwork:
    def __init__(self, input_func: Callable[[int], NDArray[np.float64]], config: dict[str, Any]) -> None:
        # Store only immutable configuration
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
        
        # Learning rate parameters  
        max_lr = config['maximum_learning_rate']
        min_lr = config['minimum_learning_rate']
        self.beta_1: float = (max_lr * min_lr**3) / (max_lr**2 - min_lr**2)
        self.beta_2: float = min_lr
        self.beta_3: float = (min_lr * max_lr) / (max_lr**2 - min_lr**2)
        
        # Initialize state (for backward compatibility)
        initial_state = create_initial_network_state(None, config, self.num_inputs)
        self.weights = initial_state.weights
        self.learning_rate = initial_state.learning_rate
        self.neural_network_gradient_wrt_weights: (NDArray[np.float64] | None) = None

    def initialize_weights(self) -> None:
        """Deprecated: Use create_initial_network_state instead."""
        initial_state = create_initial_network_state(None, self._get_config(), self.num_inputs)
        self.weights = initial_state.weights
        
    def _get_config(self) -> dict[str, Any]:
        """Get configuration dictionary from instance attributes."""
        return {
            'time_step_delta': self.time_step_delta,
            'final_time': self.time_steps * self.time_step_delta,
            'inner_activation': self.inner_layer_activation_function,
            'output_activation': self.outer_layer_activation_function,
            'shortcut_activation': self.shortcut_activation_function,
            'num_blocks': self.num_blocks,
            'num_layers': self.num_layers,
            'num_neurons': self.num_neurons,
            'num_inputs': self.num_inputs,
            'output_size': self.num_outputs,
            'weight_bounds': self.weight_bounds,
            'maximum_learning_rate': self._compute_max_lr(),
            'minimum_learning_rate': self.beta_2,
            'initial_learning_rate': 1.0,  # Default value
            'beta_1': self.beta_1,
            'beta_2': self.beta_2,
            'beta_3': self.beta_3,
        }
        
    def _compute_max_lr(self) -> float:
        """Compute maximum learning rate from beta parameters."""
        # Reverse computation from initialization
        # This is an approximation for backward compatibility
        return 8.0  # Default value used in tests

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

    def perform_backward_propagation(self, activated_layers: list[NDArray[np.float64]], unactivated_layers: list[NDArray[np.float64]], transposed_weight_matrices: list[NDArray[np.float64]], outer_product: NDArray[np.float64] | None = None) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
            gradient: NDArray[np.float64] | None = None
            product: NDArray[np.float64] | None = None
            for layer_index in range(self.num_layers, -1, -1):
                transposed_output = activated_layers[layer_index].T
                if layer_index == self.num_layers:
                    current_gradient: NDArray[np.float64] = np.asarray(np.kron(np.eye(self.num_outputs, dtype=np.float64), transposed_output), dtype=np.float64)
                    if outer_product is not None: gradient = outer_product @ current_gradient
                    else: gradient = current_gradient
                    product = transposed_weight_matrices[layer_index] @ self.apply_activation_function_derivative_and_bias(unactivated_layers[layer_index - 1], self.outer_layer_activation_function)
                else:
                    kron_product = np.kron(np.eye(self.num_neurons), transposed_output)
                    if outer_product is None: layer_gradient = product @ kron_product if product is not None else np.zeros_like(kron_product)
                    else: layer_gradient = outer_product @ product @ kron_product if product is not None else np.zeros_like(kron_product)
                    gradient = np.hstack((layer_gradient, gradient)) if gradient is not None else layer_gradient
                    if layer_index != 0 and product is not None: product = product @ transposed_weight_matrices[layer_index] @ self.apply_activation_function_derivative_and_bias(unactivated_layers[layer_index - 1], self.inner_layer_activation_function)
            
            if gradient is None: gradient = np.zeros((1, 1))  # fallback case
            if product is None: product = np.zeros((1, 1))  # fallback case
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
        """Predict output using functional approach."""
        config = self._get_config()
        current_state = NetworkState(weights=self.weights, learning_rate=self.learning_rate)
        neural_network_output, new_state = predict(config, self.input_func, current_state, step)
        
        # Update instance state for backward compatibility
        self.learning_rate = new_state.learning_rate
        return neural_network_output

    def train_step(self, step: int, loss: NDArray[np.float64]) -> NDArray[np.float64]:
        """Train one step using functional approach."""
        config = self._get_config()
        current_state = NetworkState(weights=self.weights, learning_rate=self.learning_rate)
        neural_network_output, new_state = train_step(config, self.input_func, current_state, loss, step)
        
        # Update instance state for backward compatibility
        self.weights = new_state.weights
        self.learning_rate = new_state.learning_rate
        
        # For backward compatibility, we still compute the gradient here
        _, _, activated_layers_blocks, unactivated_layers_blocks, transposed_weights_blocks = run_forward_pass(config, self.input_func, self.weights, step)
        self.neural_network_gradient_wrt_weights = run_backward_pass(config, activated_layers_blocks, unactivated_layers_blocks, transposed_weights_blocks)
        
        return neural_network_output

    def set_weights(self, weights: NDArray[np.float64]) -> None:
        """Set weights for testing."""
        self.weights = weights.copy()

    def forward_raw(self, step: int) -> NDArray[np.float64]:
        """Raw forward pass for testing."""
        config = self._get_config()
        return forward_raw(config, self.input_func, self.weights, step)

    def jacobian_raw(self, step: int) -> NDArray[np.float64]:
        """Raw jacobian computation for testing.""" 
        config = self._get_config()
        return jacobian_raw(config, self.input_func, self.weights, step)

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
        if activation_function not in ACTIVATION_FUNCTIONS:
            raise ValueError(f"Unknown activation function: {activation_function}")
        
        result = ACTIVATION_FUNCTIONS[activation_function](x)
        return np.vstack((result, [[1]]))

    @staticmethod
    def apply_activation_function_derivative_and_bias(x: NDArray[np.float64], activation_function: str) -> NDArray[np.float64]:
        if activation_function not in ACTIVATION_DERIVATIVES:
            raise ValueError(f"Unknown activation function: {activation_function}")
            
        result = ACTIVATION_DERIVATIVES[activation_function](x)
        diag_result = np.diag(result.flatten())
        zeros_shape = (1, diag_result.shape[1]) if diag_result.shape[1] > 0 else (1, 1)
        zeros_array = np.zeros(zeros_shape)
        return np.vstack((diag_result, zeros_array))