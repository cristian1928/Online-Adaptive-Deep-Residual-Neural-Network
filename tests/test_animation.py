"""
Smoke test: animation builds and returns a FuncAnimation without errors.
"""

import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, Path(__file__).resolve().parent.parent.as_posix())

from main import run_simulation
from src.io import data_manager

TEST_CONFIG = {
    "final_time": 0.1,
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
    "k1": 1,
}


def test_animation_functionality() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        orig_cwd = Path.cwd()
        orig_data_dir = data_manager.DATA_DIR
        data_manager.DATA_DIR = os.path.join(tmp, "simulation_data")

        try:
            os.chdir(tmp)
            with patch("builtins.print"):
                run_simulation(TEST_CONFIG)

            with patch("matplotlib.pyplot.show"):
                from src.visualization.plotter import animate
                animation = animate()
                assert animation is not None, "animate() should return an animation object"
        finally:
            os.chdir(orig_cwd)
            data_manager.DATA_DIR = orig_data_dir
            plt.close("all")
