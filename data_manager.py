import csv
import os
from collections import defaultdict
from typing import TYPE_CHECKING, Any, Dict, List, Optional, TextIO, Tuple

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D

from entity import Agent, Target

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


def save_state_to_csv(step: int, time: float, agents: List[Agent], target: Target) -> None:
    ensure_directory_exists(DATA_DIR)

    target_row: Dict[str, float] = {
        "Time": time,
        "Position X": target.positions[0, step - 1],
        "Position Y": target.positions[1, step - 1],
        "Position Z": target.positions[2, step - 1],
    }

    for i, agent in enumerate(agents):
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


def save_nn_to_csv(step: int, time: float, agents: List[Agent]) -> None:
    ensure_directory_exists(DATA_DIR)

    for agent in agents:
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


def configure_plot() -> None:
    plt.style.use(["science", "ieee"])
    plt.rcParams["figure.dpi"] = 100
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["axes.labelsize"] = 14
    plt.rcParams["axes.titlesize"] = 16
    plt.rcParams["xtick.labelsize"] = 12
    plt.rcParams["ytick.labelsize"] = 12
    plt.rcParams.update(
        {
            "lines.linewidth": 1.0,
            "axes.linewidth": 0.5,
            "legend.frameon": True,
            "legend.edgecolor": "black",
        }
    )


def get_simulation_data() -> Tuple[List[str], List[pd.DataFrame], pd.DataFrame]:
    csv_state_files: List[str] = [
        f
        for f in os.listdir(DATA_DIR)
        if f.endswith(STATE_DATA_SUFFIX) and not f.startswith("target")
    ]
    csv_state_files.sort()
    agent_types: List[str] = [f.replace(STATE_DATA_SUFFIX, "") for f in csv_state_files]
    agents_state_data: List[pd.DataFrame] = [
        pd.read_csv(os.path.join(DATA_DIR, f)) for f in csv_state_files
    ]
    target_state_data: pd.DataFrame = pd.read_csv(TARGET_FILE)
    return agent_types, agents_state_data, target_state_data


def get_nn_data() -> Tuple[List[str], List[pd.DataFrame]]:
    csv_nn_files: List[str] = [f for f in os.listdir(DATA_DIR) if f.endswith(NN_DATA_SUFFIX)]
    csv_nn_files.sort()
    agent_types: List[str] = [f.replace(NN_DATA_SUFFIX, "") for f in csv_nn_files]
    agents_nn_data: List[pd.DataFrame] = [
        pd.read_csv(os.path.join(DATA_DIR, f)) for f in csv_nn_files
    ]
    return agent_types, agents_nn_data


def create_plot_with_config(figsize: Tuple[int, int] = (8, 6)) -> Tuple["Figure", "Axes"]:
    fig, ax = plt.subplots(figsize=figsize)
    return fig, ax


def create_3d_plot_with_config(figsize: Tuple[int, int] = (8, 6)) -> Tuple["Figure", Axes3D]:
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection="3d")
    return fig, ax


def plot_time_series(
    ax: "Axes",
    time_data: pd.Series,
    data: pd.Series,
    xlabel: str,
    ylabel: str,
    title: Optional[str] = None,
) -> None:
    ax.plot(time_data, data)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    plt.tight_layout()


def plot_from_csv() -> None:
    configure_plot()
    agent_types, agents_state_data, target_state_data = get_simulation_data()
    nn_agent_types, agents_nn_data = get_nn_data()

    time_vals: pd.Series = agents_state_data[0]["Time"]

    fig, ax = create_plot_with_config()
    plot_data: List[Tuple[int, str, pd.Series, float]] = []
    for i, ad in enumerate(agents_state_data):
        te: pd.Series = ad["Tracking Error Norm"]
        rms: float = np.sqrt(np.mean(te**2))
        print("Tracking Error Norm (m): ", rms)
        plot_data.append((i, agent_types[i], te, rms))
    plot_data.sort(key=lambda x: x[3], reverse=True)
    for _i, agent_type, te, rms in plot_data:
        ax.plot(time_vals, te, label=f"{agent_type.title()}: RMS {rms:.4f} m")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Tracking Error Norm (m)", fontsize=16)
    ax.legend(loc="best", fontsize=14, frameon=True, edgecolor="black")
    plt.tight_layout()

    fig, ax = create_3d_plot_with_config()
    for i, pos in enumerate(agents_state_data):
        ax.plot(
            pos["Position X"].values,
            pos["Position Y"].values,
            pos["Position Z"].values,
            label=agent_types[i].title(),
            linestyle="--",
            color="blue",
        )
    ax.plot(
        target_state_data["Position X"],
        target_state_data["Position Y"],
        target_state_data["Position Z"],
        label="Target Trajectory",
        linestyle="--",
        color="black",
    )
    ax.set_xlabel("X Position (m)")
    ax.set_ylabel("Y Position (m)")
    ax.set_zlabel("Z Position (m)")
    ax.legend(loc="best", fontsize=12, frameon=True, edgecolor="black")
    ax.set_box_aspect((1, 1, 1))
    plt.tight_layout()

    for _i, nn in enumerate(agents_nn_data):
        fig, ax = create_plot_with_config()
        time_nn: pd.Series = nn["Time"]
        weight_cols: List[str] = [c for c in nn.columns if c.startswith("Weight_")]
        for col in weight_cols:
            ax.plot(time_nn, nn[col])
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Weight Value")
        plt.tight_layout()

    for _i, nn in enumerate(agents_nn_data):
        fig, ax = create_plot_with_config()
        plot_time_series(
            ax,
            nn["Time"],
            nn["Function Approximation Error Norm"],
            "Time (s)",
            "Function Approximation Error Norm",
        )

    for _i, nn in enumerate(agents_nn_data):
        fig, ax = create_plot_with_config()
        plot_time_series(
            ax,
            nn["Time"],
            nn["Learning Rate Spectral Norm"],
            "Time (s)",
            "Learning Rate Spectral Norm",
        )

    for _i, nn in enumerate(agents_nn_data):
        fig, ax = create_plot_with_config()
        plot_time_series(
            ax,
            nn["Time"],
            nn["Neural Network Output"],
            "Time (s)",
            "Neural Network Output $(m/s^2)$",
        )

    plt.show()


def animate() -> FuncAnimation:
    plt.style.use("default")
    agent_types, agents_state_data, target_state_data = get_simulation_data()

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.set_xlabel("X Position (m)")
    ax.set_ylabel("Y Position (m)")
    ax.set_zlabel("Z Position (m)")  # type: ignore[attr-defined]

    position_data: List[pd.DataFrame] = agents_state_data + [target_state_data]
    x_min, x_max = min(data["Position X"].min() for data in position_data), max(
        data["Position X"].max() for data in position_data
    )
    y_min, y_max = min(data["Position Y"].min() for data in position_data), max(
        data["Position Y"].max() for data in position_data
    )
    z_min, z_max = min(data["Position Z"].min() for data in position_data), max(
        data["Position Z"].max() for data in position_data
    )

    margin: float = 0.1
    x_range, y_range, z_range = x_max - x_min, y_max - y_min, z_max - z_min
    ax.set_xlim(x_min - margin * x_range, x_max + margin * x_range)
    ax.set_ylim(y_min - margin * y_range, y_max + margin * y_range)
    ax.set_zlim(z_min - margin * z_range, z_max + margin * z_range)  # type: ignore

    agent_lines: List[Any] = []
    agent_points: List[Any] = []
    for i, _agent_data in enumerate(agents_state_data):
        (line,) = ax.plot([], [], [], "-", label=f"{agent_types[i].title()}")
        (point,) = ax.plot([], [], [], "o", markersize=6)
        agent_lines.append(line)
        agent_points.append(point)

    (target_line,) = ax.plot([], [], [], "--", label="Target")
    (target_point,) = ax.plot([], [], [], "o", markersize=6, color="red")

    legend = ax.legend(loc="upper right", prop={"size": 7})
    legend.get_frame().set_linewidth(0.5)
    time_text = ax.text2D(0.02, 0.95, "", transform=ax.transAxes)  # type: ignore

    trail_length: int = 3500

    def update(frame: int) -> List[Any]:
        current_time: float = target_state_data["Time"][frame]
        time_text.set_text(f"Time: {current_time:.2f} s")

        start_idx: int = max(0, frame - trail_length)

        for i, data in enumerate(agents_state_data):
            agent_lines[i].set_data(
                data["Position X"][start_idx : frame + 1], data["Position Y"][start_idx : frame + 1]
            )
            agent_lines[i].set_3d_properties(data["Position Z"][start_idx : frame + 1])
            agent_points[i].set_data([data["Position X"][frame]], [data["Position Y"][frame]])
            agent_points[i].set_3d_properties([data["Position Z"][frame]])

        target_line.set_data(
            target_state_data["Position X"][start_idx : frame + 1],
            target_state_data["Position Y"][start_idx : frame + 1],
        )
        target_line.set_3d_properties(target_state_data["Position Z"][start_idx : frame + 1])  # type: ignore
        target_point.set_data(
            [target_state_data["Position X"][frame]], [target_state_data["Position Y"][frame]]
        )
        target_point.set_3d_properties([target_state_data["Position Z"][frame]])  # type: ignore

        return agent_lines + agent_points + [target_line, target_point, time_text]

    num_frames: int = len(target_state_data)

    plt.tight_layout()
    plt.show()
    return FuncAnimation(
        fig, update, frames=range(0, num_frames, max(1, num_frames // 200)), blit=False, interval=75
    )


def results() -> None:
    close_all_files()
    plot_from_csv()
