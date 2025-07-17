"""Data management utilities for RNN simulation."""

import csv
import os
from collections import defaultdict
from typing import Any, Dict, List, TextIO

import numpy as np

from ..viz.plotter import get_simulation_data

DATA_DIR: str = "simulation_data"
STATE_DATA_SUFFIX: str = "_state_data.csv"
NN_DATA_SUFFIX: str = "_nn_data.csv"
TARGET_FILE: str = f"{DATA_DIR}/target_state_data.csv"

_file_handles: Dict[str, TextIO] = {}
_csv_writers: Dict[str, csv.DictWriter[str]] = {}
_data_buffers: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
_buffer_size: int = 100


def ensure_directory_exists(directory: str) -> None:
    os.makedirs(directory, exist_ok=True)


def _get_csv_writer(file_path: str, headers: List[str], step: int) -> csv.DictWriter[str]:
    if file_path not in _file_handles:
        if step == 1 and os.path.exists(file_path):
            os.remove(file_path)
        _file_handles[file_path] = open(file_path, "w", newline="", buffering=8192)
        _csv_writers[file_path] = csv.DictWriter(_file_handles[file_path], fieldnames=headers)
        _csv_writers[file_path].writeheader()
    return _csv_writers[file_path]


def _flush_buffer(file_path: str) -> None:
    if file_path in _data_buffers and _data_buffers[file_path]:
        writer: csv.DictWriter[str] = _csv_writers[file_path]
        writer.writerows(_data_buffers[file_path])
        _file_handles[file_path].flush()
        _data_buffers[file_path].clear()


def save_state_to_csv(step: int, time: float, agents: List[Any], target: Any) -> None:
    from ..core.entity import Agent, Target

    agents_typed: List[Agent] = agents
    target_typed: Target = target

    ensure_directory_exists(DATA_DIR)

    target_row: Dict[str, float] = {
        "Time": time,
        "Position X": target_typed.positions[0, step - 1],
        "Position Y": target_typed.positions[1, step - 1],
        "Position Z": target_typed.positions[2, step - 1],
    }

    for i, agent in enumerate(agents_typed):
        tracking_error_norm: float = float(np.linalg.norm(agent.tracking_error))
        agent_type: str = getattr(agent, "agent_type", f"agent_{i}")
        state_file_path: str = f"{DATA_DIR}/{agent_type}{STATE_DATA_SUFFIX}"

        headers: List[str] = [
            "Time",
            "Position X",
            "Position Y",
            "Position Z",
            "Tracking Error Norm",
        ]
        _get_csv_writer(state_file_path, headers, step)

        row_data: Dict[str, float] = {
            "Time": time,
            "Position X": agent.positions[0, step - 1],
            "Position Y": agent.positions[1, step - 1],
            "Position Z": agent.positions[2, step - 1],
            "Tracking Error Norm": tracking_error_norm,
        }
        _data_buffers[state_file_path].append(row_data)

        if len(_data_buffers[state_file_path]) >= _buffer_size:
            _flush_buffer(state_file_path)

    target_headers: List[str] = ["Time", "Position X", "Position Y", "Position Z"]
    _get_csv_writer(TARGET_FILE, target_headers, step)
    _data_buffers[TARGET_FILE].append(target_row)

    if len(_data_buffers[TARGET_FILE]) >= _buffer_size:
        _flush_buffer(TARGET_FILE)


def save_nn_to_csv(step: int, time: float, agents: List[Any]) -> None:
    from ..core.entity import Agent

    agents_typed: List[Agent] = agents

    ensure_directory_exists(DATA_DIR)

    for agent in agents_typed:
        weights: np.ndarray = agent.neural_network.weights
        if isinstance(weights[0], (list, np.ndarray)) and len(weights[0]) == 1:
            float_weights: List[float] = [float(w[0]) for w in weights]
        else:
            float_weights = [float(w) for w in weights]

        learning_rate_matrix: np.ndarray = agent.neural_network.learning_rate[step]
        eigvals: np.ndarray = np.real(np.linalg.eigvals(learning_rate_matrix))

        nn_file_path: str = f"{DATA_DIR}/{agent.agent_type}{NN_DATA_SUFFIX}"

        headers: List[str] = [
            "Time",
            "Learning Rate Spectral Norm",
            "Function Approximation Error Norm",
            "Neural Network Output",
        ] + [f"Weight_{j + 1}" for j in range(len(float_weights))]
        _get_csv_writer(nn_file_path, headers, step)

        row_data: Dict[str, float] = {
            "Time": time,
            "Learning Rate Spectral Norm": float(np.max(eigvals)),
            "Function Approximation Error Norm": float(
                np.linalg.norm(agent.neural_network_output - agent.target.velocities[0, step - 1])
            ),
            "Neural Network Output": float(np.linalg.norm(agent.neural_network_output)),
        }
        row_data.update({f"Weight_{j + 1}": w for j, w in enumerate(float_weights)})

        _data_buffers[nn_file_path].append(row_data)

        if len(_data_buffers[nn_file_path]) >= _buffer_size:
            _flush_buffer(nn_file_path)


def close_all_files() -> None:
    for file_path in list(_data_buffers.keys()):
        _flush_buffer(file_path)

    for handle in _file_handles.values():
        handle.close()

    _file_handles.clear()
    _csv_writers.clear()
    _data_buffers.clear()


def compute_tracking_error() -> float:
    """Compute the RMS tracking error norm from the simulation data.

    Returns:
        The root mean square (RMS) tracking error norm.
    """
    agent_types, agents_state_data, target_state_data = get_simulation_data()

    if not agents_state_data:
        return 0.0

    # Use the first agent's tracking error data
    tracking_error_data = agents_state_data[0]["Tracking Error Norm"]
    rms_error = float(np.sqrt(np.mean(tracking_error_data**2)))

    return rms_error


def results() -> None:
    close_all_files()
    from ..viz.plotter import plot_from_csv

    plot_from_csv()
