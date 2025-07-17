"""
End-to-end integration test for Online-Adaptive-Deep-Residual-Neural-Network.

This test runs the entire simulation with a reference configuration and verifies:
1. The RMS tracking error norm equals 10.183181125499472 (± 1e-6)
2. Plotting and animation functionality compiles without errors

Reference configuration produces a specific RMS tracking error norm that serves
as a regression test baseline for the complete simulation pipeline.
"""

import numpy as np
import pytest
import pandas as pd
import os
import sys
import tempfile
import shutil
from pathlib import Path
from unittest.mock import patch
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for testing
import matplotlib.pyplot as plt

# Add the parent directory to the path so we can import from the main module
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from main import run_simulation
import data_manager


def test_end_to_end_simulation_tracking_error():
    """Test complete simulation with reference configuration and verify RMS tracking error norm."""
    
    # Reference configuration from the issue
    reference_config = {
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
    
    # Expected RMS tracking error norm (from issue specification)
    expected_rms_tracking_error_norm = 10.183181125499472
    tolerance = 1e-6
    
    # Create temporary directory for simulation data
    with tempfile.TemporaryDirectory() as temp_dir:
        # Save current working directory
        original_cwd = os.getcwd()
        
        # Save original DATA_DIR and update to temporary directory
        original_data_dir = data_manager.DATA_DIR
        temp_data_dir = os.path.join(temp_dir, 'simulation_data')
        data_manager.DATA_DIR = temp_data_dir
        
        try:
            # Change to temp directory for simulation
            os.chdir(temp_dir)
            
            # Run the complete simulation without plotting/animation
            with patch('builtins.print'):  # Suppress progress output during test
                run_simulation(reference_config)
            
            # Read the state data to get tracking error norm
            state_file = os.path.join(temp_data_dir, 'Residual Neural Network_state_data.csv')
            assert os.path.exists(state_file), f"State data file not found: {state_file}"
            
            state_data = pd.read_csv(state_file)
            
            # Get the RMS tracking error norm (this is what the issue refers to as "final tracking error norm")
            rms_tracking_error_norm = np.sqrt(np.mean(state_data['Tracking Error Norm']**2))
            
            # Verify the RMS tracking error norm matches expected value within tolerance
            assert abs(rms_tracking_error_norm - expected_rms_tracking_error_norm) <= tolerance, \
                f"RMS tracking error norm {rms_tracking_error_norm} does not match expected {expected_rms_tracking_error_norm} within tolerance {tolerance}"
            
            print(f"✓ RMS tracking error norm test passed: {rms_tracking_error_norm}")
            
        finally:
            # Restore original settings
            os.chdir(original_cwd)
            data_manager.DATA_DIR = original_data_dir
            # Clean up any matplotlib figures
            plt.close('all')


def test_plotting_functionality():
    """Test that plotting functionality compiles and runs without errors."""
    
    # Reference configuration (minimal simulation for testing plots)
    test_config = {
        "final_time": 0.1,  # Very short simulation for testing
        "time_step_delta": 0.01,
        "seed": 0,
        "num_states": 3,
        "control_size": 3,
        "target_initial_conditions": [40, 9, 2],
        "dynamics_type": "trophic_dynamics",
        "ID": "Test Agent",
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
    
    # Create temporary directory for simulation data
    with tempfile.TemporaryDirectory() as temp_dir:
        # Save current working directory and data directory
        original_cwd = os.getcwd()
        original_data_dir = data_manager.DATA_DIR
        temp_data_dir = os.path.join(temp_dir, 'simulation_data')
        data_manager.DATA_DIR = temp_data_dir
        
        try:
            # Disable LaTeX rendering and scienceplots style to avoid dependencies
            original_style_use = plt.style.use
            plt.style.use = lambda x: None
            
            # Also set matplotlib to not use TeX
            original_usetex = plt.rcParams.get('text.usetex', False)
            plt.rcParams['text.usetex'] = False
            
            # Change to temp directory for simulation
            os.chdir(temp_dir)
            
            # Run a short simulation to generate data
            with patch('builtins.print'):  # Suppress progress output during test
                run_simulation(test_config)
            
            # Test plotting functionality
            with patch('matplotlib.pyplot.show'):  # Suppress plot display during test
                try:
                    data_manager.plot_from_csv()
                    print("✓ Plotting functionality test passed")
                except Exception as e:
                    pytest.fail(f"Plotting functionality failed: {e}")
            
        finally:
            # Restore original settings
            os.chdir(original_cwd)
            data_manager.DATA_DIR = original_data_dir
            plt.style.use = original_style_use
            plt.rcParams['text.usetex'] = original_usetex
            # Clean up any matplotlib figures
            plt.close('all')


def test_animation_functionality():
    """Test that animation functionality compiles and runs without errors."""
    
    # Reference configuration (minimal simulation for testing animation)
    test_config = {
        "final_time": 0.1,  # Very short simulation for testing
        "time_step_delta": 0.01,
        "seed": 0,
        "num_states": 3,
        "control_size": 3,
        "target_initial_conditions": [40, 9, 2],
        "dynamics_type": "trophic_dynamics",
        "ID": "Test Agent",
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
    
    # Create temporary directory for simulation data
    with tempfile.TemporaryDirectory() as temp_dir:
        # Save current working directory and data directory
        original_cwd = os.getcwd()
        original_data_dir = data_manager.DATA_DIR
        temp_data_dir = os.path.join(temp_dir, 'simulation_data')
        data_manager.DATA_DIR = temp_data_dir
        
        try:
            # Change to temp directory for simulation
            os.chdir(temp_dir)
            
            # Run a short simulation to generate data
            with patch('builtins.print'):  # Suppress progress output during test
                run_simulation(test_config)
            
            # Test animation functionality
            with patch('matplotlib.pyplot.show'):  # Suppress animation display during test
                try:
                    animation = data_manager.animate()
                    assert animation is not None, "Animation function should return a FuncAnimation object"
                    print("✓ Animation functionality test passed")
                except Exception as e:
                    pytest.fail(f"Animation functionality failed: {e}")
            
        finally:
            # Restore original settings
            os.chdir(original_cwd)
            data_manager.DATA_DIR = original_data_dir
            # Clean up any matplotlib figures
            plt.close('all')


if __name__ == "__main__":
    test_end_to_end_simulation_tracking_error()
    test_plotting_functionality()
    test_animation_functionality()