# End-to-End Regression Test

This document describes the end-to-end regression test for the Online-Adaptive-Deep-Residual-Neural-Network.

## Overview

The test validates that the entire simulation pipeline works correctly, including:
- Forward pass computations
- Training loop execution 
- Plotting functionality
- Animation export
- Final tracking error norm validation

## Test Structure

The test is located in `tests/test_end_to_end_regression.py` and includes:

### 1. Configuration Validation Test
- **Function:** `test_config_validation`
- **Purpose:** Validates that the reference configuration has all required parameters
- **Execution Time:** ~1 second

### 2. Data Structure Test  
- **Function:** `test_simulation_data_structure`
- **Purpose:** Tests that simulation produces the expected data structure
- **Execution Time:** ~1.5 seconds (uses short simulation)

### 3. Pipeline Test
- **Function:** `test_simulation_pipeline_with_plotting`
- **Purpose:** Tests that the entire pipeline runs without errors, including plotting
- **Execution Time:** ~1.5 seconds (uses short simulation)

### 4. Main Acceptance Test
- **Function:** `test_reference_configuration_tracking_error`
- **Purpose:** Main acceptance criteria test - validates tracking error norm
- **Execution Time:** ~17 seconds (full simulation)
- **Marker:** `@pytest.mark.slow`

## Reference Configuration

The test uses a reference configuration defined in `reference_config.json`:

```json
{
    "final_time": 10,
    "time_step_delta": 0.001,
    "seed": 0,
    "num_states": 3,
    "control_size": 3,
    "target_initial_conditions": [40, 9, 2],
    "dynamics_type": "trophic_dynamics",
    "ID": "Residual Neural Network",
    "output_size": 3,
    "num_blocks": 2,
    "num_layers": 1,
    "num_neurons": 1,
    "inner_activation": "swish",
    "output_activation": "tanh",
    "shortcut_activation": "swish",
    "minimum_learning_rate": 0.01,
    "initial_learning_rate": 1,
    "maximum_learning_rate": 8,
    "weight_bounds": 2,
    "k1": 1
}
```

## Expected Results

The test validates that the tracking error norm equals `10.183181125499472` (± 1e-6).

## Running the Tests

### Run All Tests
```bash
python -m pytest tests/test_end_to_end_regression.py -v
```

### Run Only Fast Tests (exclude slow test)
```bash
python -m pytest tests/test_end_to_end_regression.py -m "not slow" -v
```

### Run Only Slow Test (main acceptance test)
```bash
python -m pytest tests/test_end_to_end_regression.py -m "slow" -v
```

## CI Integration

The test is designed to work in CI environments:

- Uses non-interactive matplotlib backend (Agg)
- Mocks plotting functions to avoid LaTeX dependencies
- Handles potential dependency issues gracefully
- Includes appropriate timeouts for long-running tests

## Troubleshooting

### LaTeX Issues
If you encounter LaTeX-related errors, the test automatically:
- Disables LaTeX rendering with `matplotlib.rcParams['text.usetex'] = False`
- Mocks plotting style setup
- Skips tests gracefully if LaTeX errors persist

### Memory Issues
For systems with limited memory, you can run only the fast tests:
```bash
python -m pytest tests/test_end_to_end_regression.py -m "not slow"
```

## Test Implementation Details

### Plotting Support
- Uses `matplotlib.use('Agg')` for headless operation
- Mocks `matplotlib.pyplot.show`, `matplotlib.pyplot.savefig`, and animation functions
- Patches `matplotlib.pyplot.style.use` to avoid LaTeX style issues

### Data Validation
- Validates simulation produces expected data structure
- Checks for required data fields: `Tracking Error Norm`, `Position X/Y/Z`, `Time`
- Ensures data types are correct (numpy arrays or pandas Series)

### Error Handling
- Gracefully handles LaTeX dependency issues
- Provides clear error messages for debugging
- Includes appropriate timeouts for long-running operations