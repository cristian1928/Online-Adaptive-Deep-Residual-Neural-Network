"""
Regression test: full simulation produces the expected RMS tracking‑error norm.
"""

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

from main import run_simulation
from src.io import data_manager

REFERENCE_CONFIG = {
    "final_time": 10,
    "time_step_delta": 0.001,
    "seed": 0,
    "num_states": 3,
    "control_size": 3,
    "dynamics_type": "trophic_dynamics",
    "ID": "Residual Neural Network",
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

EXPECTED_RMS = 8.416_460_028_855_283
TOL = 1e-6


def test_end_to_end_simulation_tracking_error() -> None:
    """Run the reference simulation and verify the RMS tracking‑error norm."""
    with tempfile.TemporaryDirectory() as tmp:
        orig_cwd = Path.cwd()
        orig_data_dir = data_manager.DATA_DIR
        data_manager.DATA_DIR = os.path.join(tmp, "simulation_data")

        try:
            os.chdir(tmp)
            with patch("builtins.print"):
                run_simulation(REFERENCE_CONFIG)

            state_file = Path(data_manager.DATA_DIR) / "Residual Neural Network_state_data.csv"
            assert state_file.exists(), f"Missing state file: {state_file}"

            df = pd.read_csv(state_file)
            rms = np.sqrt(np.mean(df["Tracking Error Norm"] ** 2))

            assert abs(rms - EXPECTED_RMS) <= TOL, (
                f"RMS {rms} differs from expected {EXPECTED_RMS} by more than {TOL}"
            )
        finally:
            os.chdir(orig_cwd)
            data_manager.DATA_DIR = orig_data_dir
            plt.close("all")
