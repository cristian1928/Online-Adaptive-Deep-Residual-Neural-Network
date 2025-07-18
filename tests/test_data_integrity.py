"""
Test data integrity across configurations to prevent data/plot mix-up.

This test ensures that:
1. Configurations with the same ID don't overwrite each other's data
2. Each configuration gets its own unique CSV files
3. Data written corresponds exactly to the configuration that generated it
4. Cross-writing cannot occur between configurations
"""

import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any
from unittest.mock import patch

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# make the package root importable
sys.path.insert(0, Path(__file__).resolve().parent.parent.as_posix())

from main import run_simulation_from_configs
from src.io import data_manager


def create_test_config(agent_id: str, k1_value: float) -> dict[str, Any]:
    """Create a test configuration with specified ID and k1 value."""
    return {
        "final_time": 1,
        "time_step_delta": 0.001,
        "seed": 0,
        "num_states": 3,
        "control_size": 3,
        "dynamics_type": "trophic_dynamics",
        "ID": agent_id,
        "output_size": 3,
        "num_blocks": 1,
        "num_layers": 1,
        "num_neurons": 1,
        "inner_activation": "swish",
        "output_activation": "tanh",
        "shortcut_activation": "swish",
        "minimum_singular_value": 0.01,
        "initial_learning_rate": 1,
        "maximum_singular_value": 8,
        "weight_bounds": 2,
        "k1": k1_value,
    }


def test_same_id_creates_unique_files() -> None:
    """Test that configurations with the same ID create separate files."""
    with tempfile.TemporaryDirectory() as tmp:
        orig_cwd = Path.cwd()
        orig_data_dir = data_manager.DATA_DIR
        data_manager.DATA_DIR = os.path.join(tmp, "simulation_data")

        try:
            os.chdir(tmp)
            
            # Create three configs with same ID but different k1 values
            configs = [
                create_test_config("TestAgent", 1.0),
                create_test_config("TestAgent", 5.0),
                create_test_config("TestAgent", 10.0),
            ]

            with patch("builtins.print"):
                run_simulation_from_configs(configs)

            # Verify separate files are created
            data_dir = Path(data_manager.DATA_DIR)
            state_files = list(data_dir.glob("*_state_data.csv"))
            state_files = [f for f in state_files if "target" not in f.name]
            
            assert len(state_files) == 3, f"Expected 3 state files, got {len(state_files)}: {[f.name for f in state_files]}"
            
            # Verify each file has data
            file_rows = []
            for state_file in sorted(state_files):
                df = pd.read_csv(state_file)
                assert len(df) > 0, f"File {state_file.name} is empty"
                file_rows.append(len(df))
                
            # All files should have approximately the same number of rows
            assert all(abs(rows - file_rows[0]) <= 2 for rows in file_rows), \
                f"File row counts vary too much: {file_rows}"

        finally:
            os.chdir(orig_cwd)
            data_manager.DATA_DIR = orig_data_dir
            plt.close("all")


def test_data_integrity_with_different_k1_values() -> None:
    """Test that data corresponds to the correct configuration parameters."""
    with tempfile.TemporaryDirectory() as tmp:
        orig_cwd = Path.cwd()
        orig_data_dir = data_manager.DATA_DIR
        data_manager.DATA_DIR = os.path.join(tmp, "simulation_data")

        try:
            os.chdir(tmp)
            
            # Test with very different k1 values to ensure data separation
            configs = [
                create_test_config("Agent", 0.1),  # Very small k1
                create_test_config("Agent", 100.0), # Very large k1
            ]

            with patch("builtins.print"):
                run_simulation_from_configs(configs)

            # Get the state files
            data_dir = Path(data_manager.DATA_DIR)
            state_files = [f for f in data_dir.glob("*_state_data.csv") if "target" not in f.name]
            assert len(state_files) == 2, f"Expected 2 files, got {len(state_files)}"

            # Read final tracking errors from both files
            final_errors = []
            for state_file in sorted(state_files):
                df = pd.read_csv(state_file)
                final_error = df["Tracking Error Norm"].iloc[-1]
                final_errors.append(final_error)

            # With very different k1 values, final tracking errors should be different
            assert abs(final_errors[0] - final_errors[1]) > 0.001, \
                f"Final errors too similar: {final_errors}. Data might be mixed up."

        finally:
            os.chdir(orig_cwd)
            data_manager.DATA_DIR = orig_data_dir
            plt.close("all")


def test_unique_agent_types_preserved() -> None:
    """Test that unique agent types are preserved (no modification for unique IDs)."""
    with tempfile.TemporaryDirectory() as tmp:
        orig_cwd = Path.cwd()
        orig_data_dir = data_manager.DATA_DIR
        data_manager.DATA_DIR = os.path.join(tmp, "simulation_data")

        try:
            os.chdir(tmp)
            
            # Create configs with unique IDs
            configs = [
                create_test_config("UniqueAgent1", 1.0),
                create_test_config("UniqueAgent2", 2.0),
                create_test_config("UniqueAgent3", 3.0),
            ]

            with patch("builtins.print"):
                run_simulation_from_configs(configs)

            # Verify files use original agent names
            data_dir = Path(data_manager.DATA_DIR)
            state_files = [f for f in data_dir.glob("*_state_data.csv") if "target" not in f.name]
            
            expected_files = {
                "UniqueAgent1_state_data.csv",
                "UniqueAgent2_state_data.csv", 
                "UniqueAgent3_state_data.csv"
            }
            actual_files = {f.name for f in state_files}
            
            assert actual_files == expected_files, \
                f"Expected files {expected_files}, got {actual_files}"

        finally:
            os.chdir(orig_cwd)
            data_manager.DATA_DIR = orig_data_dir
            plt.close("all")


def test_neural_network_files_also_unique() -> None:
    """Test that neural network CSV files are also made unique."""
    with tempfile.TemporaryDirectory() as tmp:
        orig_cwd = Path.cwd()
        orig_data_dir = data_manager.DATA_DIR
        data_manager.DATA_DIR = os.path.join(tmp, "simulation_data")

        try:
            os.chdir(tmp)
            
            # Create configs with same ID
            configs = [
                create_test_config("NNTest", 1.0),
                create_test_config("NNTest", 2.0),
            ]

            with patch("builtins.print"):
                run_simulation_from_configs(configs)

            # Verify both state and NN files are created uniquely
            data_dir = Path(data_manager.DATA_DIR)
            state_files = [f for f in data_dir.glob("*_state_data.csv") if "target" not in f.name]
            nn_files = list(data_dir.glob("*_nn_data.csv"))
            
            assert len(state_files) == 2, f"Expected 2 state files, got {len(state_files)}"
            assert len(nn_files) == 2, f"Expected 2 NN files, got {len(nn_files)}"
            
            # Verify files have data
            for nn_file in nn_files:
                df = pd.read_csv(nn_file)
                assert len(df) > 0, f"NN file {nn_file.name} is empty"

        finally:
            os.chdir(orig_cwd)
            data_manager.DATA_DIR = orig_data_dir
            plt.close("all")


def test_edge_case_with_numeric_suffix_collision() -> None:
    """Test edge case where agent ID already has numeric suffix that could collide."""
    with tempfile.TemporaryDirectory() as tmp:
        orig_cwd = Path.cwd()
        orig_data_dir = data_manager.DATA_DIR
        data_manager.DATA_DIR = os.path.join(tmp, "simulation_data")

        try:
            os.chdir(tmp)
            
            # Create configs that could cause suffix collision
            configs = [
                create_test_config("Agent", 1.0),      # Will become "Agent"
                create_test_config("Agent", 2.0),      # Will become "Agent_1" 
                create_test_config("Agent_1", 3.0),    # Already has "_1" suffix
            ]

            with patch("builtins.print"):
                run_simulation_from_configs(configs)

            # All should create separate files without collision
            data_dir = Path(data_manager.DATA_DIR)
            state_files = [f for f in data_dir.glob("*_state_data.csv") if "target" not in f.name]
            
            assert len(state_files) == 3, f"Expected 3 files, got {len(state_files)}: {[f.name for f in state_files]}"
            
            # Verify all files have different names
            file_names = {f.name for f in state_files}
            assert len(file_names) == 3, f"File name collision detected: {file_names}"

        finally:
            os.chdir(orig_cwd)
            data_manager.DATA_DIR = orig_data_dir
            plt.close("all")


if __name__ == "__main__":
    test_same_id_creates_unique_files()
    test_data_integrity_with_different_k1_values()
    test_unique_agent_types_preserved()
    test_neural_network_files_also_unique()
    test_edge_case_with_numeric_suffix_collision()
    print("All data integrity tests passed!")