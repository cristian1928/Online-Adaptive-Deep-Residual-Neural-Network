import csv
import os
from collections import defaultdict
from csv import DictWriter
from typing import TYPE_CHECKING, Any, Dict, List, TextIO

import numpy as np

if TYPE_CHECKING:
    from src.core.entity import Agent, Target
    # For mypy, use the modern, specific type hint
    CSVDictWriter = DictWriter[Any]
else:
    # For older Python runtimes, use the non-subscriptable class
    CSVDictWriter = DictWriter

# Constants for data management
DATA_DIR = 'simulation_data'
STATE_DATA_SUFFIX = '_state_data.csv'
NN_DATA_SUFFIX = '_nn_data.csv'
TARGET_FILE = f'{DATA_DIR}/target_state_data.csv'

# Global file handles and data buffers for efficient writing
_file_handles: Dict[str, TextIO] = {}
_csv_writers: Dict[str, CSVDictWriter] = {}
_data_buffers: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
_buffer_size: int = 100

def ensure_directory_exists(directory: str) -> None:
    """Create directory if it doesn't exist."""
    os.makedirs(directory, exist_ok=True)

def _get_csv_writer(file_path: str, headers: List[str], step: int) -> CSVDictWriter:
    """Get or create a CSV writer for the given file path."""
    if file_path not in _file_handles:
        if step == 1 and os.path.exists(file_path): 
            os.remove(file_path)
        _file_handles[file_path] = open(file_path, 'w', newline='', buffering=8192)
        _csv_writers[file_path] = csv.DictWriter(_file_handles[file_path], fieldnames=headers)
        _csv_writers[file_path].writeheader()
    return _csv_writers[file_path]

def _flush_buffer(file_path: str) -> None:
    """Flush buffered data to file."""
    if file_path in _data_buffers and _data_buffers[file_path]:
        writer = _csv_writers[file_path]
        writer.writerows(_data_buffers[file_path])
        _file_handles[file_path].flush()
        _data_buffers[file_path].clear()

def save_state_to_csv(step: int, time: float, agents: List["Agent"], target: "Target") -> None:
    """Save agent and target state data to CSV files."""
    ensure_directory_exists(DATA_DIR)

    target_row: Dict[str, Any] = {
        'Time': time, 
        'Position X': target.positions[0, step - 1], 
        'Position Y': target.positions[1, step - 1], 
        'Position Z': target.positions[2, step - 1]
    }

    # Process agents
    for i, agent in enumerate(agents):
        tracking_error_norm = np.linalg.norm(agent.tracking_error)
        agent_type = getattr(agent, 'agent_type', f'agent_{i}')
        state_file_path = f'{DATA_DIR}/{agent_type}{STATE_DATA_SUFFIX}'

        # Initialize writer if needed
        headers = ['Time', 'Position X', 'Position Y', 'Position Z', 'Tracking Error Norm']
        _get_csv_writer(state_file_path, headers, step)

        # Buffer the data instead of writing immediately
        row_data: Dict[str, Any] = {
            'Time': time,
            'Position X': agent.positions[0, step - 1],
            'Position Y': agent.positions[1, step - 1],
            'Position Z': agent.positions[2, step - 1],
            'Tracking Error Norm': tracking_error_norm,
        }
        _data_buffers[state_file_path].append(row_data)

        if len(_data_buffers[state_file_path]) >= _buffer_size: 
            _flush_buffer(state_file_path)

    target_headers = ['Time', 'Position X', 'Position Y', 'Position Z']
    _get_csv_writer(TARGET_FILE, target_headers, step)
    _data_buffers[TARGET_FILE].append(target_row)

    if len(_data_buffers[TARGET_FILE]) >= _buffer_size:  
        _flush_buffer(TARGET_FILE)

def save_nn_to_csv(step: int, time: float, agents: List["Agent"]) -> None:
    """Save neural network data to CSV files."""
    ensure_directory_exists(DATA_DIR)

    for agent in agents:
        weights = agent.neural_network.weights
        if isinstance(weights[0], (list, np.ndarray)) and len(weights[0]) == 1: 
            float_weights = [float(w[0]) for w in weights]
        else: 
            float_weights = [float(w) for w in weights]

        learning_rate_matrix = agent.neural_network.learning_rate[step]

        nn_file_path = f'{DATA_DIR}/{agent.agent_type}{NN_DATA_SUFFIX}'

        headers = [
            'Time', 
            'Learning Rate Spectral Norm', 
            'Function Approximation Error Norm', 
            'Neural Network Output',
        ] + [f'Weight_{j + 1}' for j in range(len(float_weights))]
        
        _get_csv_writer(nn_file_path, headers, step)
        
        row_data: Dict[str, Any] = {
            'Time': time,
            'Learning Rate Spectral Norm': np.linalg.norm(learning_rate_matrix, 2),
            'Function Approximation Error Norm': np.linalg.norm(agent.neural_network_output - agent.target.velocities[:, step - 1]),
            'Neural Network Output': np.linalg.norm(agent.neural_network_output)
        }
        row_data.update({f'Weight_{j + 1}': w for j, w in enumerate(float_weights)})

        _data_buffers[nn_file_path].append(row_data)

        if len(_data_buffers[nn_file_path]) >= _buffer_size: 
            _flush_buffer(nn_file_path)

def close_all_files() -> None:
    """Close all open file handles and flush remaining data."""
    for file_path in list(_data_buffers.keys()): 
        _flush_buffer(file_path)

    for handle in _file_handles.values(): 
        handle.close()

    _file_handles.clear()
    _csv_writers.clear()
    _data_buffers.clear()