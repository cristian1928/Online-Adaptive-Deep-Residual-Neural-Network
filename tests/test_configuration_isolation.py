"""
Test to ensure that data/plot mix-up across configurations cannot occur.

This test verifies that when the same configuration is run alone vs with other
configurations, it produces identical results, ensuring no cross-contamination.
"""

import json
import os
import tempfile
from pathlib import Path
from unittest.mock import patch
import sys

import numpy as np
import pandas as pd

# Add the repo to Python path
repo_path = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_path))

from main import run_simulation_from_configs, load_configurations
from src.io import data_manager


def test_configuration_isolation() -> None:
    """Test that configurations produce identical results regardless of what other configurations are present."""
    
    # Test configurations
    common_config = {
        "final_time": 2,  # Short simulation for faster testing
        "time_step_delta": 0.01,
        "seed": 42,
        "num_states": 3,
        "control_size": 3,
        "dynamics_type": "trophic_dynamics",
        "output_size": 3,
        "inner_activation": "swish",
        "output_activation": "tanh",
        "shortcut_activation": "swish",
        "minimum_singular_value": 0.01,
        "initial_learning_rate": 1,
        "maximum_singular_value": 8,
        "weight_bounds": 4,
        "k1": 1
    }
    
    resnet_config = {**common_config, "ID": "resnet", "num_blocks": 2, "num_layers": 2, "num_neurons": 2}
    shallow_config = {**common_config, "ID": "shallow", "num_blocks": 1, "num_layers": 1, "num_neurons": 1}
    deep_config = {**common_config, "ID": "deep", "num_blocks": 3, "num_layers": 3, "num_neurons": 3}
    
    with tempfile.TemporaryDirectory() as tmp_dir:
        # Save original values
        orig_cwd = Path.cwd()
        orig_data_dir = data_manager.DATA_DIR
        
        try:
            os.chdir(tmp_dir)
            data_manager.DATA_DIR = os.path.join(tmp_dir, "simulation_data")
            
            # Test 1: Run ResNet alone
            with patch("builtins.print"):
                run_simulation_from_configs([resnet_config])
            
            # Read and save ResNet alone data
            resnet_alone_state = pd.read_csv(os.path.join(data_manager.DATA_DIR, "resnet_state_data.csv"))
            resnet_alone_nn = pd.read_csv(os.path.join(data_manager.DATA_DIR, "resnet_nn_data.csv"))
            
            # Clear data for next test
            os.system(f"rm -rf {data_manager.DATA_DIR}")
            
            # Test 2: Run ResNet with other configurations
            with patch("builtins.print"):
                run_simulation_from_configs([resnet_config, shallow_config, deep_config])
            
            # Read ResNet data when run with others
            resnet_with_others_state = pd.read_csv(os.path.join(data_manager.DATA_DIR, "resnet_state_data.csv"))
            resnet_with_others_nn = pd.read_csv(os.path.join(data_manager.DATA_DIR, "resnet_nn_data.csv"))
            
            # Verify that ResNet data is identical in both cases
            pd.testing.assert_frame_equal(resnet_alone_state, resnet_with_others_state, 
                                        check_exact=False, rtol=1e-10, atol=1e-12)
            pd.testing.assert_frame_equal(resnet_alone_nn, resnet_with_others_nn,
                                        check_exact=False, rtol=1e-10, atol=1e-12)
            
            # Test 3: Verify different configurations produce different data
            shallow_state = pd.read_csv(os.path.join(data_manager.DATA_DIR, "shallow_state_data.csv"))
            deep_state = pd.read_csv(os.path.join(data_manager.DATA_DIR, "deep_state_data.csv"))
            
            # Ensure different configurations produce different results
            assert not resnet_alone_state.equals(shallow_state), "ResNet and Shallow should produce different results"
            assert not resnet_alone_state.equals(deep_state), "ResNet and Deep should produce different results"
            assert not shallow_state.equals(deep_state), "Shallow and Deep should produce different results"
            
            print("✓ All configuration isolation tests passed!")
            
        finally:
            os.chdir(orig_cwd)
            data_manager.DATA_DIR = orig_data_dir


def test_neural_network_deterministic_initialization() -> None:
    """Test that neural networks are initialized deterministically based on configuration."""
    
    from src.core.neural_network import NeuralNetwork
    
    config1 = {
        'time_step_delta': 0.01,
        'final_time': 5,
        'inner_activation': 'swish',
        'output_activation': 'tanh',
        'shortcut_activation': 'swish',
        'num_blocks': 2,
        'num_layers': 2,
        'num_neurons': 2,
        'output_size': 3,
        'weight_bounds': 4,
        'minimum_singular_value': 0.01,
        'maximum_singular_value': 8,
        'initial_learning_rate': 1,
        'seed': 42,
        'ID': 'test_config'
    }
    
    config2 = config1.copy()
    config2['ID'] = 'different_config'  # Different ID should produce different weights
    
    def input_func(step: int) -> np.ndarray:
        return np.array([1.0, 2.0, 3.0])
    
    # Test 1: Same config should produce same weights regardless of global random state
    np.random.seed(100)
    nn1a = NeuralNetwork(input_func, config1)
    
    np.random.seed(200)  # Different global seed
    nn1b = NeuralNetwork(input_func, config1)
    
    assert np.allclose(nn1a.weights, nn1b.weights), "Same config should produce same weights regardless of global random state"
    
    # Test 2: Different configs should produce different weights
    nn2 = NeuralNetwork(input_func, config2)
    assert not np.allclose(nn1a.weights, nn2.weights), "Different configs should produce different weights"
    
    print("✓ Neural network deterministic initialization tests passed!")


if __name__ == "__main__":
    test_neural_network_deterministic_initialization()
    test_configuration_isolation()
    print("✓ All tests passed! Data mix-up issue is resolved.")