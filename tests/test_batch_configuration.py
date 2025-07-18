"""
Test batch configuration loading functionality.
"""

import json
import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# make the package root importable
sys.path.insert(0, Path(__file__).resolve().parent.parent.as_posix())

from main import load_configurations, run_simulation_from_configs
from src.io import data_manager

REFERENCE_CONFIG_1 = {
    "final_time": 5,
    "time_step_delta": 0.001,
    "seed": 0,
    "num_states": 3,
    "control_size": 3,
    "dynamics_type": "trophic_dynamics",
    "ID": "Agent_1",
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
    "k1": 1,
}

REFERENCE_CONFIG_2 = {
    "final_time": 5,
    "time_step_delta": 0.001,
    "seed": 0,
    "num_states": 3,
    "control_size": 3,
    "dynamics_type": "trophic_dynamics",
    "ID": "Agent_2",
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
    "k1": 1,
}

# Baseline config for testing baseline functionality
BASELINE_CONFIG = {
    "final_time": 5,
    "time_step_delta": 0.001,
    "seed": 0,
    "num_states": 3,
    "control_size": 3,
    "dynamics_type": "trophic_dynamics",
}

# Minimal agent configs that will be merged with baseline
MINIMAL_CONFIG_1 = {
    "ID": "Agent_1",
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
    "k1": 1,
}

MINIMAL_CONFIG_2 = {
    "ID": "Agent_2",
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
    "k1": 1,
}


def test_load_configurations_single_config() -> None:
    """Test loading a single configuration file."""
    with tempfile.TemporaryDirectory() as tmp:
        orig_cwd = Path.cwd()
        try:
            os.chdir(tmp)
            
            # Create configurations directory with single config
            config_dir = Path("configurations")
            config_dir.mkdir()
            
            config_file = config_dir / "config1.json"
            with open(config_file, "w") as f:
                json.dump(REFERENCE_CONFIG_1, f)
            
            configs = load_configurations()
            
            assert len(configs) == 1
            assert configs[0]["ID"] == "Agent_1"
            
        finally:
            os.chdir(orig_cwd)


def test_load_configurations_multiple_configs() -> None:
    """Test loading multiple configuration files."""
    with tempfile.TemporaryDirectory() as tmp:
        orig_cwd = Path.cwd()
        try:
            os.chdir(tmp)
            
            # Create configurations directory with multiple configs
            config_dir = Path("configurations")
            config_dir.mkdir()
            
            config_file1 = config_dir / "config1.json"
            with open(config_file1, "w") as f:
                json.dump(REFERENCE_CONFIG_1, f)
                
            config_file2 = config_dir / "config2.json"
            with open(config_file2, "w") as f:
                json.dump(REFERENCE_CONFIG_2, f)
            
            configs = load_configurations()
            
            assert len(configs) == 2
            # Should be sorted by filename
            assert configs[0]["ID"] == "Agent_1"
            assert configs[1]["ID"] == "Agent_2"
            
        finally:
            os.chdir(orig_cwd)


def test_load_configurations_with_baseline() -> None:
    """Test loading configurations with baseline config merging."""
    with tempfile.TemporaryDirectory() as tmp:
        orig_cwd = Path.cwd()
        try:
            os.chdir(tmp)
            
            # Create configurations directory with baseline and specific configs
            config_dir = Path("configurations")
            config_dir.mkdir()
            
            # Create baseline config
            baseline_file = config_dir / "config_common.json"
            with open(baseline_file, "w") as f:
                json.dump(BASELINE_CONFIG, f)
            
            # Create minimal configs that should be merged with baseline
            config_file1 = config_dir / "config1.json"
            with open(config_file1, "w") as f:
                json.dump(MINIMAL_CONFIG_1, f)
                
            config_file2 = config_dir / "config2.json"
            with open(config_file2, "w") as f:
                json.dump(MINIMAL_CONFIG_2, f)
            
            configs = load_configurations()
            
            assert len(configs) == 2
            
            # Verify configs have merged baseline parameters
            for config in configs:
                assert config["final_time"] == 5
                assert config["time_step_delta"] == 0.001
                assert config["seed"] == 0
                assert config["num_states"] == 3
                assert config["control_size"] == 3
                assert config["dynamics_type"] == "trophic_dynamics"
            
            # Verify specific agent parameters
            assert configs[0]["ID"] == "Agent_1"
            assert configs[1]["ID"] == "Agent_2"
            
        finally:
            os.chdir(orig_cwd)


def test_load_configurations_no_baseline() -> None:
    """Test loading configurations without baseline config (backward compatibility)."""
    with tempfile.TemporaryDirectory() as tmp:
        orig_cwd = Path.cwd()
        try:
            os.chdir(tmp)
            
            # Create configurations directory with full configs (no baseline)
            config_dir = Path("configurations")
            config_dir.mkdir()
            
            config_file1 = config_dir / "config1.json"
            with open(config_file1, "w") as f:
                json.dump(REFERENCE_CONFIG_1, f)
                
            config_file2 = config_dir / "config2.json"
            with open(config_file2, "w") as f:
                json.dump(REFERENCE_CONFIG_2, f)
            
            configs = load_configurations()
            
            assert len(configs) == 2
            assert configs[0]["ID"] == "Agent_1"
            assert configs[1]["ID"] == "Agent_2"
            # Verify all parameters are present even without baseline
            assert configs[0]["final_time"] == 5
            assert configs[0]["time_step_delta"] == 0.001
            
        finally:
            os.chdir(orig_cwd)


def test_batch_simulation_multiple_agents() -> None:
    """Test running batch simulation with multiple agents."""
    with tempfile.TemporaryDirectory() as tmp:
        orig_cwd = Path.cwd()
        orig_data_dir = data_manager.DATA_DIR
        data_manager.DATA_DIR = os.path.join(tmp, "simulation_data")

        try:
            os.chdir(tmp)
            configs = [REFERENCE_CONFIG_1, REFERENCE_CONFIG_2]
            
            with patch("builtins.print"):
                run_simulation_from_configs(configs)

            # Verify both agents created output files
            data_dir = Path(data_manager.DATA_DIR)
            
            agent1_file = data_dir / "Agent_1_state_data.csv"
            agent2_file = data_dir / "Agent_2_state_data.csv"
            
            assert agent1_file.exists(), f"Missing agent 1 state file: {agent1_file}"
            assert agent2_file.exists(), f"Missing agent 2 state file: {agent2_file}"
            
            # Verify files have data
            df1 = pd.read_csv(agent1_file)
            df2 = pd.read_csv(agent2_file)
            
            assert len(df1) > 0, "Agent 1 state file is empty"
            assert len(df2) > 0, "Agent 2 state file is empty"
            
        finally:
            os.chdir(orig_cwd)
            data_manager.DATA_DIR = orig_data_dir
            plt.close("all")