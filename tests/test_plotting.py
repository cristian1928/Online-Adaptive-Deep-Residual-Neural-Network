"""
Smoke test: generating static plots should complete without raising.
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
    "dynamics_type": "trophic_dynamics",
    "ID": "Test Agent",
    "output_size": 3,
    "num_blocks": 2,
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


def test_plotting_functionality() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        orig_cwd = Path.cwd()
        orig_data_dir = data_manager.DATA_DIR
        data_manager.DATA_DIR = os.path.join(tmp, "simulation_data")

        # temporarily disable external styles/LaTeX
        saved_style_use = plt.style.use
        saved_usetex = plt.rcParams.get("text.usetex", False)

        try:
            plt.style.use = lambda *_: None
            plt.rcParams["text.usetex"] = False

            os.chdir(tmp)
            with patch("builtins.print"):
                run_simulation(TEST_CONFIG)

            with patch("matplotlib.pyplot.show"):
                from src.visualization.plotter import plot_from_csv
                plot_from_csv()  # expect no exception
        finally:
            os.chdir(orig_cwd)
            data_manager.DATA_DIR = orig_data_dir
            plt.style.use = saved_style_use
            plt.rcParams["text.usetex"] = saved_usetex
            plt.close("all")
