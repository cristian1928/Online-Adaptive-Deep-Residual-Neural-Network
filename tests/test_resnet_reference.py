import os
import sys
from typing import Any, Dict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from neural_network import NeuralNetwork


def test_resnet_reference_forward_and_gradient() -> None:
    config: Dict[str, Any] = {
        "time_step_delta": 0.001,
        "final_time": 1.0,
        "num_blocks": 2,
        "num_layers": 1,
        "num_neurons": 2,
        "output_size": 3,
        "inner_activation": "swish",
        "output_activation": "tanh",
        "shortcut_activation": "tanh",
        "minimum_learning_rate": 0.01,
        "initial_learning_rate": 1.0,
        "maximum_learning_rate": 8.0,
        "weight_bounds": 4.0,
    }

    def input_func(step: int) -> np.ndarray:
        return np.array([0.1, -0.5, 0.25])

    nn: NeuralNetwork = NeuralNetwork(input_func, config)

    theta_0: np.ndarray = np.array(
        [
            -0.9,
            -0.8,
            -0.7,
            -0.6,
            -0.5,
            -0.4,
            -0.3,
            -0.2,
            -0.1,
            0.0,
            0.1,
            0.2,
            0.3,
            0.4,
            0.5,
            0.6,
            0.7,
        ],
        dtype=float,
    )

    theta_1: np.ndarray = np.array(
        [0.8, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0, -0.1, -0.2, -0.3, -0.4, -0.5, -0.6],
        dtype=float,
    )

    theta_2: np.ndarray = np.array(
        [0.5, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, -0.7, 0.2, 0.1, 0.0, -0.1],
        dtype=float,
    )

    weights: np.ndarray = np.hstack([theta_0, theta_1, theta_2]).reshape(-1, 1)
    nn.set_weights(weights)

    expected_y: np.ndarray = np.array([1.2625721047, -0.0712802368, -0.9033769290])

    dtheta_0: np.ndarray = np.array(
        [
            [
                0.01391869,
                -0.06959344,
                0.03479672,
                0.13918687,
                0.04230649,
                -0.21153243,
                0.10576621,
                0.42306486,
                -0.67511316,
                -0.1933715,
                1.55502079,
                -0.26981733,
                -0.07728331,
                0.62148329,
                -0.17602256,
                -0.05041784,
                0.40544127,
            ],
            [
                0.00275993,
                -0.01379965,
                0.00689983,
                0.02759931,
                0.00453883,
                -0.02269415,
                0.01134707,
                0.0453883,
                0.11529236,
                0.03302299,
                -0.26555847,
                -0.26885923,
                -0.07700889,
                0.61927647,
                0.10107152,
                0.02894974,
                -0.23280292,
            ],
            [
                0.03295834,
                -0.1647917,
                0.08239585,
                0.32958339,
                0.04405756,
                -0.22028778,
                0.11014389,
                0.44057555,
                0.10079872,
                0.0288716,
                -0.23217456,
                0.08686455,
                0.02488046,
                -0.20007931,
                -0.36723297,
                -0.10518591,
                0.84586544,
            ],
        ]
    )

    dtheta_1: np.ndarray = np.array(
        [
            [
                2.69637215e-03,
                5.09352193e-03,
                7.32661544e-03,
                1.89299027e-02,
                -1.75682548e-02,
                -3.31869215e-02,
                -4.77366771e-02,
                -1.23338076e-01,
                1.41548587e00,
                9.48483112e-01,
                1.61268551e00,
                5.81826397e-01,
                3.89867906e-01,
                6.62884047e-01,
                3.81378642e-01,
                2.55552676e-01,
                4.34510739e-01,
            ],
            [
                -8.12275219e-04,
                -1.53441046e-03,
                -2.20712417e-03,
                -5.70258479e-03,
                -3.91943072e-03,
                -7.40391356e-03,
                -1.06499252e-02,
                -2.75163953e-02,
                -2.14965928e-01,
                -1.44043509e-01,
                -2.44914092e-01,
                5.58931314e-01,
                3.74526460e-01,
                6.36799315e-01,
                -1.92717812e-01,
                -1.29135581e-01,
                -2.19566461e-01,
            ],
            [
                -1.32487565e-02,
                -2.50272692e-02,
                -3.59996834e-02,
                -9.30130028e-02,
                -4.79530504e-02,
                -9.05846448e-02,
                -1.30298616e-01,
                -3.36654781e-01,
                3.38981852e-02,
                2.27143604e-02,
                3.86207403e-02,
                2.95897858e-02,
                1.98274054e-02,
                3.37121125e-02,
                8.98450450e-01,
                6.02030085e-01,
                1.02361885e00,
            ],
        ]
    )

    dtheta_2: np.ndarray = np.array(
        [
            [
                0.22425388,
                -0.14022421,
                -0.42525419,
                0.62159484,
                0.20283896,
                -0.12683362,
                -0.38464492,
                0.56223619,
                0.33467503,
                0.25087782,
                1.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
            ],
            [
                0.12814508,
                -0.08012812,
                -0.2430024,
                0.35519705,
                -0.23664545,
                0.14797256,
                0.44875241,
                -0.65594222,
                0.0,
                0.0,
                0.0,
                0.33467503,
                0.25087782,
                1.0,
                0.0,
                0.0,
                0.0,
            ],
            [
                0.03203627,
                -0.02003203,
                -0.0607506,
                0.08879926,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.33467503,
                0.25087782,
                1.0,
            ],
        ]
    )

    expected_dtheta: np.ndarray = np.hstack([dtheta_0, dtheta_1, dtheta_2])

    step: int = 0

    actual_y: np.ndarray = nn.forward_raw(step)
    actual_y_flat: np.ndarray = actual_y.flatten()

    actual_dtheta: np.ndarray = nn.jacobian_raw(step)

    np.testing.assert_allclose(
        actual_y_flat,
        expected_y,
        atol=1e-6,
        rtol=1e-6,
        err_msg=f"Forward pass output mismatch.\nExpected: {expected_y}\nActual: {actual_y_flat}",
    )

    np.testing.assert_allclose(
        actual_dtheta,
        expected_dtheta,
        atol=1e-6,
        rtol=1e-6,
        err_msg=(
            f"Gradient computation mismatch.\nExpected shape: {expected_dtheta.shape}\n"
            f"Actual shape: {actual_dtheta.shape}"
        ),
    )

    print("✓ ResNet reference test passed - forward pass and gradients match expected values")


if __name__ == "__main__":
    test_resnet_reference_forward_and_gradient()
