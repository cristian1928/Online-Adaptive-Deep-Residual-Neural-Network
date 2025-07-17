"""Plotting and animation utilities for RNN simulation."""

import os
from typing import TYPE_CHECKING, Any, List, Optional, Tuple

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D

DATA_DIR: str = "simulation_data"
STATE_DATA_SUFFIX: str = "_state_data.csv"
NN_DATA_SUFFIX: str = "_nn_data.csv"
TARGET_FILE: str = f"{DATA_DIR}/target_state_data.csv"


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
    ax.set_zlabel("Z Position (m)")  # 3D axes have set_zlabel
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
    ax.set_zlabel("Z Position (m)")  # 3D axes have set_zlabel

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
    ax.set_zlim(z_min - margin * z_range, z_max + margin * z_range)  # 3D axes have set_zlim

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
    time_text = ax.text2D(0.02, 0.95, "", transform=ax.transAxes)  # 3D axes have text2D

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
        target_line.set_3d_properties(
            target_state_data["Position Z"][start_idx : frame + 1]
        )  # 3D line method
        target_point.set_data(
            [target_state_data["Position X"][frame]], [target_state_data["Position Y"][frame]]
        )
        target_point.set_3d_properties([target_state_data["Position Z"][frame]])  # 3D point method

        return agent_lines + agent_points + [target_line, target_point, time_text]

    num_frames: int = len(target_state_data)

    plt.tight_layout()
    plt.show()
    return FuncAnimation(
        fig, update, frames=range(0, num_frames, max(1, num_frames // 200)), blit=False, interval=75
    )


def results() -> None:
    plot_from_csv()
