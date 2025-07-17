"""Integration test for tracking error norm validation."""

import os
import sys
from typing import Any, Dict

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Now we can import the actual test code
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import main


@pytest.mark.integration
def test_tracking_error_norm() -> None:
    """Test that the simulation produces the expected tracking error norm.

    This test uses the exact configuration from the issue specification
    and validates that the tracking error norm is 4.657555165934869 ± 1e-6.
    """
    config: Dict[str, Any] = {
        "final_time": 30,
        "time_step_delta": 0.001,
        "seed": 0,
        "num_states": 3,
        "control_size": 3,
        "target_initial_conditions": [40, 9, 2],
        "dynamics_type": "trophic_dynamics",
        "ID": "Residual Neural Network",
        "output_size": 3,
        "num_blocks": 4,
        "num_layers": 2,
        "num_neurons": 2,
        "inner_activation": "swish",
        "output_activation": "tanh",
        "shortcut_activation": "swish",
        "minimum_learning_rate": 0.01,
        "initial_learning_rate": 1,
        "maximum_learning_rate": 8,
        "weight_bounds": 4,
        "k1": 1,
    }

    # Run the pipeline in headless mode
    error_norm = main.run_pipeline(config, headless=True)

    # Validate the expected tracking error norm
    expected_norm = 4.657555165934869
    tolerance = 1e-6

    assert abs(error_norm - expected_norm) < tolerance, (
        f"Expected tracking error norm {expected_norm} ± {tolerance}, " f"but got {error_norm}"
    )

    print(f"✓ Integration test passed - tracking error norm: {error_norm}")


if __name__ == "__main__":
    test_tracking_error_norm()
