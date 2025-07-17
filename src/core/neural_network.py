from ..simulation.integrate import integrate_step
import numpy as np
import numpy.typing as npt
from typing import Dict, List, Tuple, Any, Callable, Optional, Union

class NeuralNetwork:
    def __init__(self, input_func: Callable[[int], npt.NDArray[np.floating[Any]]], config: Dict[str, Any]) -> None:
        self.time_step_delta: float = config['time_step_delta']
        self.time_steps: int = int(config['final_time'] / self.time_step_delta)
        self.input_func: Callable[[int], npt.NDArray[np.floating[Any]]] = input_func
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
        self.neural_network_gradient_wrt_weights: Optional[npt.NDArray[np.floating[Any]]] = None
        self.beta_1: float =  (config['maximum_learning_rate'] * config['minimum_learning_rate']**3) / (config['maximum_learning_rate']**2 - config['minimum_learning_rate']**2)
        self.beta_2: float = config['minimum_learning_rate']
        self.beta_3: float = (config['minimum_learning_rate'] * config['maximum_learning_rate']) / (config['maximum_learning_rate']**2 - config['minimum_learning_rate']**2)    

        self.learning_rate: npt.NDArray[np.floating[Any]] = (config['initial_learning_rate'] * np.eye(np.size(self.weights)))[None, :, :].repeat(self.time_steps, axis=0)

    def initialize_weights(self) -> None:
        activation_to_variance: Dict[str, int] = {'tanh': 1, 'sigmoid': 1, 'identity': 1, 'swish': 2, 'relu': 2, 'leaky_relu': 2}
        inner_layer_variance: int = activation_to_variance[self.inner_layer_activation_function]
        output_layer_variance: int = activation_to_variance[self.outer_layer_activation_function]
        weights: List[npt.NDArray[np.floating[Any]]] = []
        for block in range(self.num_blocks + 1):
            input_size = self.num_inputs if block == 0 else self.num_outputs
            weights.append(self.generate_initialized_weights(input_size, self.num_neurons, inner_layer_variance))
            weights.extend(self.generate_initialized_weights(self.num_neurons, self.num_neurons, inner_layer_variance) for _ in range(self.num_layers - 1))
            weights.append(self.generate_initialized_weights(self.num_neurons, self.num_outputs, output_layer_variance))
        self.weights: npt.NDArray[np.floating[Any]] = np.vstack(weights)

    def generate_initialized_weights(self, input_size: int, output_size: int, variance_factor: int) -> npt.NDArray[np.floating[Any]]:
        variance: float = variance_factor / input_size  # Applies either Xavier (1/input) or He (2/input) initialization
        return np.random.normal(0, np.sqrt(variance), output_size * (input_size + 1)).reshape(-1, 1) # input_size + 1 accounts for bias term

    def get_input_with_bias(self, step: int) -> npt.NDArray[np.floating[Any]]: 
        return np.append(self.input_func(step), 1).reshape(-1, 1)

    def construct_transposed_weight_matrices(self, weight_index: int) -> Tuple[int, List[npt.NDArray[np.floating[Any]]]]:
        weight_matrices: List[npt.NDArray[np.floating[Any]]] = []
        biased_input_size: int = self.num_inputs + 1 if weight_index == 0 else self.num_outputs + 1
        biased_neuron_size: int = self.num_neurons + 1
        layer_shapes: List[Tuple[int, int]] = [(biased_input_size, self.num_neurons)] + [(biased_neuron_size, self.num_neurons)] * (self.num_layers - 1) + [(biased_neuron_size, self.num_outputs)]
        for rows, cols in layer_shapes:
            matrix = np.array(self.weights[weight_index:weight_index + rows * cols]).reshape(rows, cols, order='F')
            weight_matrices.append(matrix.T)
            weight_index += rows * cols
        return weight_index, weight_matrices

    def perform_forward_propagation(self, transposed_weight_matrices: List[npt.NDArray[np.floating[Any]]], input_with_bias: npt.NDArray[np.floating[Any]]) -> Tuple[List[npt.NDArray[np.floating[Any]]], List[npt.NDArray[np.floating[Any]]]]:
        activated_layers: List[npt.NDArray[np.floating[Any]]] = [input_with_bias]
        unactivated_layers: List[npt.NDArray[np.floating[Any]]] = []
        for layer_index in range(self.num_layers + 1):
            unactivated_output = transposed_weight_matrices[layer_index] @ activated_layers[-1]
            unactivated_layers.append(unactivated_output)
            if layer_index != self.num_layers:
                activation_function = self.outer_layer_activation_function if layer_index == self.num_layers - 1 else self.inner_layer_activation_function
                activated_layers.append(self.apply_activation_function_and_bias(unactivated_output, activation_function))
        return activated_layers, unactivated_layers

    def perform_backward_propagation(self, activated_layers: List[npt.NDArray[np.floating[Any]]], unactivated_layers: List[npt.NDArray[np.floating[Any]]], transposed_weight_matrices: List[npt.NDArray[np.floating[Any]]], outer_product: Optional[npt.NDArray[np.floating[Any]]] = None) -> Tuple[npt.NDArray[np.floating[Any]], npt.NDArray[np.floating[Any]]]:
        gradient: Optional[npt.NDArray[np.floating[Any]]] = None
        product: Optional[npt.NDArray[np.floating[Any]]] = None
        for layer_index in range(self.num_layers, -1, -1):
            transposed_output = activated_layers[layer_index].T
            if layer_index == self.num_layers:
                gradient = np.kron(np.eye(self.num_outputs), transposed_output)
                if outer_product is not None: 
                    gradient = outer_product @ gradient
                product = transposed_weight_matrices[layer_index] @ self.apply_activation_function_derivative_and_bias(unactivated_layers[layer_index - 1], self.outer_layer_activation_function)
            else:
                kron_product = np.kron(np.eye(self.num_neurons), transposed_output)
                if product is None:
                    raise ValueError("Product not initialized in backward propagation")
                layer_gradient = product @ kron_product if outer_product is None else outer_product @ product @ kron_product
                gradient = np.hstack((layer_gradient, gradient)) if gradient is not None else layer_gradient
                if layer_index != 0: 
                    product = product @ transposed_weight_matrices[layer_index] @ self.apply_activation_function_derivative_and_bias(unactivated_layers[layer_index - 1], self.inner_layer_activation_function)
        
        if gradient is None:
            raise ValueError("Gradient calculation failed")
        if product is None:
            raise ValueError("Product calculation failed")
        return gradient, product

    def _run_forward_pass(self, step: int) -> Tuple[int, npt.NDArray[np.floating[Any]], List[List[npt.NDArray[np.floating[Any]]]], List[List[npt.NDArray[np.floating[Any]]]], List[List[npt.NDArray[np.floating[Any]]]]]:
        weight_index = 0
        neural_network_output: npt.NDArray[np.floating[Any]] = np.zeros(self.num_outputs).reshape(-1, 1)
        activated_layers_blocks: List[List[npt.NDArray[np.floating[Any]]]] = [[] for _ in range(self.num_blocks + 1)]
        unactivated_layers_blocks: List[List[npt.NDArray[np.floating[Any]]]] = [[] for _ in range(self.num_blocks + 1)]
        transposed_weights_blocks: List[List[npt.NDArray[np.floating[Any]]]] = [[] for _ in range(self.num_blocks + 1)]
        for block_index in range(self.num_blocks + 1):
            weight_index, transposed_weights_blocks[block_index] = self.construct_transposed_weight_matrices(weight_index)
            input_data = self.get_input_with_bias(step) if block_index == 0 else self.apply_activation_function_and_bias(neural_network_output, self.shortcut_activation_function)
            activated_layers_blocks[block_index], unactivated_layers_blocks[block_index] = self.perform_forward_propagation(transposed_weights_blocks[block_index], input_data)
            neural_network_output += unactivated_layers_blocks[block_index][-1]
        return weight_index, neural_network_output, activated_layers_blocks, unactivated_layers_blocks, transposed_weights_blocks

    def _run_backward_pass(self, activated_layers_blocks: List[List[npt.NDArray[np.floating[Any]]]], unactivated_layers_blocks: List[List[npt.NDArray[np.floating[Any]]]], transposed_weights_blocks: List[List[npt.NDArray[np.floating[Any]]]]) -> npt.NDArray[np.floating[Any]]:
        outer_product: Optional[npt.NDArray[np.floating[Any]]] = None
        total_gradient: Optional[npt.NDArray[np.floating[Any]]] = None
        for block_index in range(self.num_blocks, -1, -1):
            current_outer_product = outer_product if block_index < self.num_blocks else None
            block_gradient, inner_product = self.perform_backward_propagation(activated_layers_blocks[block_index], unactivated_layers_blocks[block_index], transposed_weights_blocks[block_index], current_outer_product)
            if block_index == self.num_blocks: 
                total_gradient = block_gradient
            else: 
                total_gradient = np.hstack((block_gradient, total_gradient)) if total_gradient is not None else block_gradient
            if block_index > 0:
                block_output = sum(unactivated_layers_blocks[i][-1] for i in range(block_index))
                # Ensure block_output is a numpy array
                if not isinstance(block_output, np.ndarray):
                    block_output = np.array(block_output)
                preactivation_derivative = self.apply_activation_function_derivative_and_bias(block_output, self.shortcut_activation_function)
                update_term = inner_product @ transposed_weights_blocks[block_index][0] @ preactivation_derivative
                if block_index == self.num_blocks: 
                    outer_product = np.eye(self.num_outputs) + update_term
                else: 
                    outer_product = outer_product @ (np.eye(self.num_outputs) + update_term) if outer_product is not None else (np.eye(self.num_outputs) + update_term)
        
        if total_gradient is None:
            raise ValueError("Total gradient calculation failed")
        return total_gradient

    def predict(self, step: int) -> npt.NDArray[np.floating[Any]]:
        self.learning_rate[step] = self.learning_rate[step - 1]
        _, neural_network_output, _, _, _ = self._run_forward_pass(step)
        return neural_network_output

    def train_step(self, step: int, loss: npt.NDArray[np.floating[Any]]) -> npt.NDArray[np.floating[Any]]:
        _, neural_network_output, activated_layers_blocks, unactivated_layers_blocks, transposed_weights_blocks = self._run_forward_pass(step)
        total_gradient = self._run_backward_pass(activated_layers_blocks, unactivated_layers_blocks, transposed_weights_blocks)
        self.neural_network_gradient_wrt_weights = total_gradient
        self.update_neural_network_weights(step, loss)
        self.update_learning_rate(step)
        return neural_network_output

    def set_weights(self, weights: npt.NDArray[np.floating[Any]]) -> None:
        self.weights = weights.copy()

    def forward_raw(self, step: int) -> npt.NDArray[np.floating[Any]]:
        _, neural_network_output, _, _, _ = self._run_forward_pass(step)
        return neural_network_output

    def jacobian_raw(self, step: int) -> npt.NDArray[np.floating[Any]]:
        _, neural_network_output, activated_layers_blocks, unactivated_layers_blocks, transposed_weights_blocks = self._run_forward_pass(step)
        total_gradient = self._run_backward_pass(activated_layers_blocks, unactivated_layers_blocks, transposed_weights_blocks)
        return total_gradient

    def update_learning_rate(self, step: int) -> None:
        def learning_rate_deriv(t: float, gamma: npt.NDArray[np.floating[Any]]) -> npt.NDArray[np.floating[Any]]:
            if self.neural_network_gradient_wrt_weights is None:
                raise ValueError("Neural network gradient not computed")
            normalized_neural_network_gradient_wrt_weights =  self.neural_network_gradient_wrt_weights / (1.0 + np.linalg.norm(self.neural_network_gradient_wrt_weights.T @ self.neural_network_gradient_wrt_weights, 'fro')**2)
            mat = normalized_neural_network_gradient_wrt_weights @ gamma
            result: npt.NDArray[np.floating[Any]] = - mat.T @ mat + (self.beta_1 * np.eye(gamma.shape[0])) + (self.beta_2 * gamma) - (self.beta_3 * gamma @ gamma)
            return result
        self.learning_rate[step] = integrate_step(self.learning_rate[step - 1], step, self.time_step_delta, learning_rate_deriv)

    def update_neural_network_weights(self, step: int, loss: npt.NDArray[np.floating[Any]]) -> None:
        def weights_deriv(t: float, weights: npt.NDArray[np.floating[Any]]) -> npt.NDArray[np.floating[Any]]:
            if self.neural_network_gradient_wrt_weights is None:
                raise ValueError("Neural network gradient not computed")
            weight_derivative = self.learning_rate[step] @ (self.neural_network_gradient_wrt_weights.T @ loss)
            projected_weights = self.proj(weight_derivative, weights, self.weight_bounds)
            return projected_weights
        self.weights = integrate_step(self.weights, step, self.time_step_delta, weights_deriv)

    def proj(self, Theta: npt.NDArray[np.floating[Any]], thetaHat: npt.NDArray[np.floating[Any]], thetaBar: float) -> npt.NDArray[np.floating[Any]]:
        max_term = max(0.0, np.dot(thetaHat.T, thetaHat) - thetaBar**2)
        dot_term = np.dot(thetaHat.T, Theta)
        numerator = max_term**2 * (dot_term + np.sqrt(dot_term**2 + 1.0)) * thetaHat
        denominator = 2.0 * (1.0 + 2.0 * thetaBar)**2 * thetaBar**2
        result: npt.NDArray[np.floating[Any]] = Theta - (numerator / denominator)
        return result

    @staticmethod
    def apply_activation_function_and_bias(x: npt.NDArray[np.floating[Any]], activation_function: str) -> npt.NDArray[np.floating[Any]]:
        result: npt.NDArray[np.floating[Any]]
        if activation_function == 'tanh': result = np.tanh(x)
        elif activation_function == 'swish': result = x * (1.0 / (1.0 + np.exp(-x)))
        elif activation_function == 'identity': result = x
        elif activation_function == 'relu': result = np.maximum(0, x)
        elif activation_function == 'sigmoid': result = 1 / (1 + np.exp(-x))
        elif activation_function == 'leaky_relu': result = np.where(x > 0, x, 0.01 * x)
        else: raise ValueError(f"Unknown activation function: {activation_function}")
        bias_row: npt.NDArray[np.floating[Any]] = np.array([[1.0]])
        return np.vstack((result, bias_row))

    @staticmethod
    def apply_activation_function_derivative_and_bias(x: npt.NDArray[np.floating[Any]], activation_function: str) -> npt.NDArray[np.floating[Any]]:
        result: npt.NDArray[np.floating[Any]]
        if activation_function == 'tanh': result = 1 - np.tanh(x)**2
        elif activation_function == 'swish':
            sigmoid = 1.0 / (1.0 + np.exp(-x))
            swish = x * sigmoid
            result = swish + sigmoid * (1 - swish)
        elif activation_function == 'identity': result = np.ones_like(x)
        elif activation_function == 'relu': result = (x > 0).astype(float)
        elif activation_function == 'sigmoid':
            sigmoid = 1 / (1 + np.exp(-x))
            result = sigmoid * (1 - sigmoid)
        elif activation_function == 'leaky_relu': result = np.where(x > 0, 1.0, 0.01)
        else: raise ValueError(f"Unknown activation function: {activation_function}")
        diag_result = np.diag(result.flatten())
        zero_row: npt.NDArray[np.floating[Any]] = np.zeros((1, diag_result.shape[1]))
        return np.vstack((diag_result, zero_row))