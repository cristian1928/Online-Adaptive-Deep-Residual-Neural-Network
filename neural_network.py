from typing import Any, Callable, Dict, List, NamedTuple, Optional, Tuple

import numpy as np

from integrate import integrate_step


class ForwardPassResult(NamedTuple):
    """Result of a forward pass through the neural network."""

    weight_index: int
    neural_network_output: np.ndarray
    activated_layers_blocks: List[List[np.ndarray]]
    unactivated_layers_blocks: List[List[np.ndarray]]
    transposed_weights_blocks: List[List[np.ndarray]]


class NeuralNetwork:
    def __init__(self, input_func: Callable[[int], np.ndarray], config: Dict[str, Any]) -> None:
        self.time_step_delta: float = config["time_step_delta"]
        self.time_steps: int = int(config["final_time"] / self.time_step_delta)
        self.input_func: Callable[[int], np.ndarray] = input_func
        self.inner_layer_activation_function: str = config["inner_activation"]
        self.outer_layer_activation_function: str = config["output_activation"]
        self.shortcut_activation_function: str = config["shortcut_activation"]
        self.num_blocks: int = config["num_blocks"]
        self.num_layers: int = config["num_layers"]
        self.num_neurons: int = config["num_neurons"]
        self.num_inputs: int = input_func(1).shape[0]
        self.num_outputs: int = config["output_size"]
        self.weight_bounds: float = config["weight_bounds"]
        self.initialize_weights()
        self.neural_network_gradient_wrt_weights: Optional[np.ndarray] = None
        self.beta_1: float = (
            config["maximum_learning_rate"] * config["minimum_learning_rate"] ** 3
        ) / (config["maximum_learning_rate"] ** 2 - config["minimum_learning_rate"] ** 2)
        self.beta_2: float = config["minimum_learning_rate"]
        self.beta_3: float = (config["minimum_learning_rate"] * config["maximum_learning_rate"]) / (
            config["maximum_learning_rate"] ** 2 - config["minimum_learning_rate"] ** 2
        )
        self.learning_rate: np.ndarray = (
            config["initial_learning_rate"] * np.eye(np.size(self.weights))
        )[None, :, :].repeat(self.time_steps, axis=0)
        self.current_step: int = 0

    def initialize_weights(self) -> None:
        activation_to_variance: Dict[str, int] = {
            "tanh": 1,
            "sigmoid": 1,
            "identity": 1,
            "swish": 2,
            "relu": 2,
            "leaky_relu": 2,
        }
        inner_layer_variance: int = activation_to_variance[self.inner_layer_activation_function]
        output_layer_variance: int = activation_to_variance[self.outer_layer_activation_function]
        weights: List[np.ndarray] = []

        for block in range(self.num_blocks + 1):
            input_size: int = self.num_inputs if block == 0 else self.num_outputs
            weights.append(
                self.generate_initialized_weights(
                    input_size, self.num_neurons, inner_layer_variance
                )
            )
            weights.extend(
                self.generate_initialized_weights(
                    self.num_neurons, self.num_neurons, inner_layer_variance
                )
                for _ in range(self.num_layers - 1)
            )
            weights.append(
                self.generate_initialized_weights(
                    self.num_neurons, self.num_outputs, output_layer_variance
                )
            )

        self.weights: np.ndarray = np.vstack(weights)

    def generate_initialized_weights(
        self, input_size: int, output_size: int, variance_factor: int
    ) -> np.ndarray:
        variance: float = variance_factor / input_size
        return np.random.normal(0, np.sqrt(variance), output_size * (input_size + 1)).reshape(-1, 1)

    def get_input_with_bias(self, step: int) -> np.ndarray:
        return np.append(self.input_func(step), 1).reshape(-1, 1)

    def construct_transposed_weight_matrices(
        self, weight_index: int
    ) -> Tuple[int, List[np.ndarray]]:
        weight_matrices: List[np.ndarray] = []
        biased_input_size: int = self.num_inputs + 1 if weight_index == 0 else self.num_outputs + 1
        biased_neuron_size: int = self.num_neurons + 1
        layer_shapes: List[Tuple[int, int]] = (
            [(biased_input_size, self.num_neurons)]
            + [(biased_neuron_size, self.num_neurons)] * (self.num_layers - 1)
            + [(biased_neuron_size, self.num_outputs)]
        )

        for rows, cols in layer_shapes:
            matrix: np.ndarray = np.array(
                self.weights[weight_index : weight_index + rows * cols]
            ).reshape(rows, cols, order="F")
            weight_matrices.append(matrix.T)
            weight_index += rows * cols

        return weight_index, weight_matrices

    def perform_forward_propagation(
        self, transposed_weight_matrices: List[np.ndarray], input_with_bias: np.ndarray
    ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        activated_layers: List[np.ndarray] = [input_with_bias]
        unactivated_layers: List[np.ndarray] = []

        for layer_index in range(self.num_layers + 1):
            unactivated_output: np.ndarray = (
                transposed_weight_matrices[layer_index] @ activated_layers[-1]
            )
            unactivated_layers.append(unactivated_output)

            if layer_index != self.num_layers:
                activation_function: str = (
                    self.outer_layer_activation_function
                    if layer_index == self.num_layers - 1
                    else self.inner_layer_activation_function
                )
                activated_layers.append(
                    self.apply_activation_function_and_bias(unactivated_output, activation_function)
                )

        return activated_layers, unactivated_layers

    def perform_backward_propagation(
        self,
        activated_layers: List[np.ndarray],
        unactivated_layers: List[np.ndarray],
        transposed_weight_matrices: List[np.ndarray],
        outer_product: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        gradient: np.ndarray = np.array([])
        product: np.ndarray = np.array([])

        for layer_index in range(self.num_layers, -1, -1):
            transposed_output: np.ndarray = activated_layers[layer_index].T

            if layer_index == self.num_layers:
                gradient = np.kron(np.eye(self.num_outputs), transposed_output)
                if outer_product is not None:
                    gradient = outer_product @ gradient
                product = transposed_weight_matrices[
                    layer_index
                ] @ self.apply_activation_function_derivative_and_bias(
                    unactivated_layers[layer_index - 1], self.outer_layer_activation_function
                )
            else:
                kron_product: np.ndarray = np.kron(np.eye(self.num_neurons), transposed_output)
                layer_gradient: np.ndarray = (
                    product @ kron_product
                    if outer_product is None
                    else outer_product @ product @ kron_product
                )
                gradient = np.hstack((layer_gradient, gradient))

                if layer_index != 0:
                    product = (
                        product
                        @ transposed_weight_matrices[layer_index]
                        @ self.apply_activation_function_derivative_and_bias(
                            unactivated_layers[layer_index - 1],
                            self.inner_layer_activation_function,
                        )
                    )

        return gradient, product

    def _run_forward_pass(self, step: int) -> ForwardPassResult:
        weight_index: int = 0
        neural_network_output: np.ndarray = np.zeros(self.num_outputs).reshape(-1, 1)
        activated_layers_blocks: List[List[np.ndarray]] = [[] for _ in range(self.num_blocks + 1)]
        unactivated_layers_blocks: List[List[np.ndarray]] = [[] for _ in range(self.num_blocks + 1)]
        transposed_weights_blocks: List[List[np.ndarray]] = [[] for _ in range(self.num_blocks + 1)]

        for block_index in range(self.num_blocks + 1):
            weight_index, transposed_weights_blocks[block_index] = (
                self.construct_transposed_weight_matrices(weight_index)
            )
            input_data: np.ndarray = (
                self.get_input_with_bias(step)
                if block_index == 0
                else self.apply_activation_function_and_bias(
                    neural_network_output, self.shortcut_activation_function
                )
            )
            activated_layers_blocks[block_index], unactivated_layers_blocks[block_index] = (
                self.perform_forward_propagation(transposed_weights_blocks[block_index], input_data)
            )
            neural_network_output += unactivated_layers_blocks[block_index][-1]

        return ForwardPassResult(
            weight_index=weight_index,
            neural_network_output=neural_network_output,
            activated_layers_blocks=activated_layers_blocks,
            unactivated_layers_blocks=unactivated_layers_blocks,
            transposed_weights_blocks=transposed_weights_blocks,
        )

    def _run_backward_pass(
        self,
        activated_layers_blocks: List[List[np.ndarray]],
        unactivated_layers_blocks: List[List[np.ndarray]],
        transposed_weights_blocks: List[List[np.ndarray]],
    ) -> np.ndarray:
        outer_product: Optional[np.ndarray] = None
        total_gradient: np.ndarray = np.array([])

        for block_index in range(self.num_blocks, -1, -1):
            current_outer_product: Optional[np.ndarray] = (
                outer_product if block_index < self.num_blocks else None
            )
            block_gradient, inner_product = self.perform_backward_propagation(
                activated_layers_blocks[block_index],
                unactivated_layers_blocks[block_index],
                transposed_weights_blocks[block_index],
                current_outer_product,
            )

            if block_index == self.num_blocks:
                total_gradient = block_gradient
            else:
                total_gradient = np.hstack((block_gradient, total_gradient))

            if block_index > 0:
                block_output = unactivated_layers_blocks[0][-1]
                for i in range(1, block_index):
                    block_output = block_output + unactivated_layers_blocks[i][-1]
                preactivation_derivative: np.ndarray = (
                    self.apply_activation_function_derivative_and_bias(
                        block_output, self.shortcut_activation_function
                    )
                )
                update_term: np.ndarray = (
                    inner_product
                    @ transposed_weights_blocks[block_index][0]
                    @ preactivation_derivative
                )

                if block_index == self.num_blocks:
                    outer_product = np.eye(self.num_outputs) + update_term
                else:
                    outer_product = outer_product @ (np.eye(self.num_outputs) + update_term)

        return total_gradient

    def predict(self, step: int) -> np.ndarray:
        self.learning_rate[step] = self.learning_rate[step - 1]
        forward_result = self._run_forward_pass(step)
        return forward_result.neural_network_output

    def train_step(self, step: int, loss_gradient: np.ndarray) -> np.ndarray:
        forward_result = self._run_forward_pass(step)
        total_gradient = self._run_backward_pass(
            forward_result.activated_layers_blocks,
            forward_result.unactivated_layers_blocks,
            forward_result.transposed_weights_blocks,
        )
        self.neural_network_gradient_wrt_weights = total_gradient
        self.update_neural_network_weights(step, loss_gradient)
        self.update_learning_rate(step)
        return forward_result.neural_network_output

    def set_weights(self, weights: np.ndarray) -> None:
        self.weights = weights.copy()

    def forward_raw(self, step: int) -> np.ndarray:
        forward_result = self._run_forward_pass(step)
        return forward_result.neural_network_output

    def jacobian_raw(self, step: int) -> np.ndarray:
        forward_result = self._run_forward_pass(step)
        total_gradient = self._run_backward_pass(
            forward_result.activated_layers_blocks,
            forward_result.unactivated_layers_blocks,
            forward_result.transposed_weights_blocks,
        )
        return total_gradient

    def _normalize_gradient(self, gradient: np.ndarray) -> np.ndarray:
        """Normalize the gradient using the Frobenius norm."""
        frobenius_norm_squared = np.linalg.norm(gradient.T @ gradient, "fro") ** 2
        normalization_factor = 1.0 + frobenius_norm_squared
        return gradient / normalization_factor

    def _compute_learning_rate_derivative(self, gamma: np.ndarray) -> np.ndarray:
        """Compute the derivative for learning rate update."""
        if self.neural_network_gradient_wrt_weights is None:
            return np.zeros_like(gamma)

        normalized_gradient = self._normalize_gradient(self.neural_network_gradient_wrt_weights)
        matrix_product = normalized_gradient @ gamma

        quadratic_term = -matrix_product.T @ matrix_product
        linear_term = self.beta_1 * np.eye(gamma.shape[0]) + self.beta_2 * gamma
        nonlinear_term = -self.beta_3 * gamma @ gamma

        return quadratic_term + linear_term + nonlinear_term

    def update_learning_rate(self, step: int) -> None:
        self.learning_rate[step] = integrate_step(
            self.learning_rate[step - 1],
            step,
            self.time_step_delta,
            lambda t, gamma: self._compute_learning_rate_derivative(gamma),
        )

    def _compute_weight_derivative(self, loss_gradient: np.ndarray) -> np.ndarray:
        """Compute the derivative for weight update."""
        if self.neural_network_gradient_wrt_weights is None:
            return np.zeros_like(self.weights)

        gradient_loss_product = self.neural_network_gradient_wrt_weights.T @ loss_gradient
        weight_derivative = self.learning_rate[self.current_step] @ gradient_loss_product
        projected_weights = self.proj(weight_derivative, self.weights, self.weight_bounds)
        return projected_weights

    def update_neural_network_weights(self, step: int, loss_gradient: np.ndarray) -> None:
        self.current_step = step  # Store current step for use in derivative computation
        self.weights = integrate_step(
            self.weights,
            step,
            self.time_step_delta,
            lambda t, weights: self._compute_weight_derivative(loss_gradient),
        )

    def proj(self, theta: np.ndarray, theta_hat: np.ndarray, theta_bar: float) -> np.ndarray:
        max_term: float = max(0.0, np.dot(theta_hat.T, theta_hat) - theta_bar**2)
        dot_term: float = np.dot(theta_hat.T, theta)
        numerator: np.ndarray = max_term**2 * (dot_term + np.sqrt(dot_term**2 + 1.0)) * theta_hat
        denominator: float = 2.0 * (1.0 + 2.0 * theta_bar) ** 2 * theta_bar**2
        return theta - (numerator / denominator)

    @staticmethod
    def apply_activation_function_and_bias(x: np.ndarray, activation_function: str) -> np.ndarray:
        if activation_function == "tanh":
            result = np.tanh(x)
        elif activation_function == "swish":
            result = x * (1.0 / (1.0 + np.exp(-x)))
        elif activation_function == "identity":
            result = x
        elif activation_function == "relu":
            result = np.maximum(0, x)
        elif activation_function == "sigmoid":
            result = 1 / (1 + np.exp(-x))
        elif activation_function == "leaky_relu":
            result = np.where(x > 0, x, 0.01 * x)
        return np.vstack((result, [[1]]))

    @staticmethod
    def apply_activation_function_derivative_and_bias(
        x: np.ndarray, activation_function: str
    ) -> np.ndarray:
        if activation_function == "tanh":
            result = 1 - np.tanh(x) ** 2
        elif activation_function == "swish":
            sigmoid = 1.0 / (1.0 + np.exp(-x))
            swish = x * sigmoid
            result = swish + sigmoid * (1 - swish)
        elif activation_function == "identity":
            result = np.ones_like(x)
        elif activation_function == "relu":
            result = (x > 0).astype(float)
        elif activation_function == "sigmoid":
            sigmoid = 1 / (1 + np.exp(-x))
            result = sigmoid * (1 - sigmoid)
        elif activation_function == "leaky_relu":
            result = np.where(x > 0, 1, 0.01)

        diag_result: np.ndarray = np.diag(result.flatten())
        return np.vstack((diag_result, np.zeros(diag_result.shape[1])))
