"""Data management utilities for simulation data."""

import csv
import os
from collections import defaultdict
from typing import TYPE_CHECKING, Any, Dict, List, Optional, TextIO, Tuple

import pandas as pd

if TYPE_CHECKING:
    from ..core.entity import Agent, Target

DATA_DIR: str = "simulation_data"
STATE_DATA_SUFFIX: str = "_state_data.csv"
NN_DATA_SUFFIX: str = "_nn_data.csv"
TARGET_FILE: str = f"{DATA_DIR}/target_state_data.csv"

_file_handles: Dict[str, TextIO] = {}
_csv_writers: Dict[str, csv.DictWriter] = {}
_data_buffers: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
_buffer_size: int = 100


def ensure_directory_exists(directory: str) -> None:
    """Create directory if it doesn't exist."""
    os.makedirs(directory, exist_ok=True)


def _get_csv_writer(file_path: str, headers: List[str], step: int) -> csv.DictWriter:
    """Get or create CSV writer for the given file path."""
    if file_path not in _file_handles:
        if step == 1 and os.path.exists(file_path):
            os.remove(file_path)
        _file_handles[file_path] = open(file_path, "w", newline="", buffering=8192)
        _csv_writers[file_path] = csv.DictWriter(_file_handles[file_path], fieldnames=headers)
        _csv_writers[file_path].writeheader()
    return _csv_writers[file_path]


def _flush_buffer(file_path: str) -> None:
    """Flush the data buffer for the given file path."""
    if file_path in _data_buffers and _data_buffers[file_path]:
        writer: csv.DictWriter = _csv_writers[file_path]
        writer.writerows(_data_buffers[file_path])
        _file_handles[file_path].flush()
        _data_buffers[file_path].clear()


def save_state_to_csv(step: int, time: float, agents: List["Agent"], target: "Target") -> None:
    """Save agent and target state data to CSV files."""
    ensure_directory_exists(DATA_DIR)
    
    # Save agent data
    for agent in agents:
        agent_file: str = f"{DATA_DIR}/{agent.agent_type.replace(' ', '_')}{STATE_DATA_SUFFIX}"
        
        headers: List[str] = [
            "Step",
            "Time",
            "Position X",
            "Position Y", 
            "Position Z",
            "Velocity X",
            "Velocity Y",
            "Velocity Z",
            "Tracking Error X",
            "Tracking Error Y",
            "Tracking Error Z",
            "Tracking Error Norm",
            "Control Output X",
            "Control Output Y",
            "Control Output Z",
            "Control Output Norm",
            "Neural Network Output X",
            "Neural Network Output Y",
            "Neural Network Output Z",
            "Neural Network Output Norm",
        ]
        
        writer: csv.DictWriter = _get_csv_writer(agent_file, headers, step)
        
        row_data: Dict[str, Any] = {
            "Step": step,
            "Time": time,
            "Position X": agent.positions[0, step - 1],
            "Position Y": agent.positions[1, step - 1],
            "Position Z": agent.positions[2, step - 1],
            "Velocity X": agent.velocities[0, step - 1],
            "Velocity Y": agent.velocities[1, step - 1],
            "Velocity Z": agent.velocities[2, step - 1],
            "Tracking Error X": agent.tracking_error[0],
            "Tracking Error Y": agent.tracking_error[1],
            "Tracking Error Z": agent.tracking_error[2],
            "Tracking Error Norm": float(sum(agent.tracking_error**2) ** 0.5),
            "Control Output X": agent.control_output[0],
            "Control Output Y": agent.control_output[1],
            "Control Output Z": agent.control_output[2],
            "Control Output Norm": float(sum(agent.control_output**2) ** 0.5),
            "Neural Network Output X": agent.neural_network_output[0],
            "Neural Network Output Y": agent.neural_network_output[1],
            "Neural Network Output Z": agent.neural_network_output[2],
            "Neural Network Output Norm": float(sum(agent.neural_network_output**2) ** 0.5),
        }
        
        _data_buffers[agent_file].append(row_data)
        if len(_data_buffers[agent_file]) >= _buffer_size:
            _flush_buffer(agent_file)
    
    # Save target data
    target_headers: List[str] = [
        "Step",
        "Time",
        "Position X",
        "Position Y",
        "Position Z",
        "Velocity X",
        "Velocity Y",
        "Velocity Z",
    ]
    
    writer = _get_csv_writer(TARGET_FILE, target_headers, step)
    
    target_row_data: Dict[str, Any] = {
        "Step": step,
        "Time": time,
        "Position X": target.positions[0, step - 1],
        "Position Y": target.positions[1, step - 1],
        "Position Z": target.positions[2, step - 1],
        "Velocity X": target.velocities[0, step - 1],
        "Velocity Y": target.velocities[1, step - 1],
        "Velocity Z": target.velocities[2, step - 1],
    }
    
    _data_buffers[TARGET_FILE].append(target_row_data)
    if len(_data_buffers[TARGET_FILE]) >= _buffer_size:
        _flush_buffer(TARGET_FILE)


def save_nn_to_csv(step: int, time: float, agents: List["Agent"]) -> None:
    """Save neural network data to CSV files."""
    ensure_directory_exists(DATA_DIR)
    
    for agent in agents:
        nn_file: str = f"{DATA_DIR}/{agent.agent_type.replace(' ', '_')}{NN_DATA_SUFFIX}"
        
        headers: List[str] = ["Step", "Time"] + [f"Weight_{i}" for i in range(agent.neural_network.weights.size)]
        
        writer: csv.DictWriter = _get_csv_writer(nn_file, headers, step)
        
        row_data: Dict[str, Any] = {
            "Step": step,
            "Time": time,
        }
        
        for i, weight in enumerate(agent.neural_network.weights.flatten()):
            row_data[f"Weight_{i}"] = weight
        
        _data_buffers[nn_file].append(row_data)
        if len(_data_buffers[nn_file]) >= _buffer_size:
            _flush_buffer(nn_file)


def close_all_files() -> None:
    """Close all open CSV files and flush remaining data."""
    for file_path in list(_data_buffers.keys()):
        if _data_buffers[file_path]:
            _flush_buffer(file_path)
    
    for file_handle in _file_handles.values():
        if not file_handle.closed:
            file_handle.close()
    
    _file_handles.clear()
    _csv_writers.clear()
    _data_buffers.clear()


def get_simulation_data() -> Tuple[List[str], List[pd.DataFrame], pd.DataFrame]:
    """Load simulation data from CSV files."""
    ensure_directory_exists(DATA_DIR)
    
    # Find all agent state files
    agent_files: List[str] = []
    agent_types: List[str] = []
    
    for file in os.listdir(DATA_DIR):
        if file.endswith(STATE_DATA_SUFFIX) and file != os.path.basename(TARGET_FILE):
            agent_files.append(os.path.join(DATA_DIR, file))
            agent_type = file.replace(STATE_DATA_SUFFIX, "").replace("_", " ")
            agent_types.append(agent_type)
    
    # Load agent data
    agents_state_data: List[pd.DataFrame] = []
    for file in agent_files:
        try:
            df = pd.read_csv(file)
            agents_state_data.append(df)
        except (FileNotFoundError, pd.errors.EmptyDataError):
            print(f"Warning: Could not load {file}")
    
    # Load target data
    try:
        target_state_data = pd.read_csv(TARGET_FILE)
    except (FileNotFoundError, pd.errors.EmptyDataError):
        print(f"Warning: Could not load {TARGET_FILE}")
        target_state_data = pd.DataFrame()
    
    return agent_types, agents_state_data, target_state_data


def get_nn_data() -> Tuple[List[str], List[pd.DataFrame]]:
    """Load neural network data from CSV files."""
    ensure_directory_exists(DATA_DIR)
    
    # Find all neural network files
    nn_files: List[str] = []
    agent_types: List[str] = []
    
    for file in os.listdir(DATA_DIR):
        if file.endswith(NN_DATA_SUFFIX):
            nn_files.append(os.path.join(DATA_DIR, file))
            agent_type = file.replace(NN_DATA_SUFFIX, "").replace("_", " ")
            agent_types.append(agent_type)
    
    # Load neural network data
    agents_nn_data: List[pd.DataFrame] = []
    for file in nn_files:
        try:
            df = pd.read_csv(file)
            agents_nn_data.append(df)
        except (FileNotFoundError, pd.errors.EmptyDataError):
            print(f"Warning: Could not load {file}")
    
    return agent_types, agents_nn_data