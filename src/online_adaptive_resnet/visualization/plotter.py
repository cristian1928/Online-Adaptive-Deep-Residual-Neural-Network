"""Plotting and visualization utilities."""

from typing import TYPE_CHECKING, Any, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scienceplots
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D

from ..io.data_manager import get_simulation_data, get_nn_data, close_all_files

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure


def configure_plot() -> None:
    """Configure matplotlib for scientific plotting."""
    try:
        plt.style.use(["science", "ieee"])
    except OSError:
        plt.style.use("default")
    plt.rcParams["font.size"] = 12
    plt.rcParams["axes.linewidth"] = 1.5
    plt.rcParams["legend.frameon"] = True
    plt.rcParams["legend.framealpha"] = 1.0
    plt.rcParams["legend.fancybox"] = True
    plt.rcParams["legend.edgecolor"] = "black"


def create_plot_with_config(figsize: Tuple[int, int] = (8, 6)) -> Tuple["Figure", "Axes"]:
    """Create a matplotlib figure with scientific styling."""
    return plt.subplots(figsize=figsize)


def create_3d_plot_with_config(figsize: Tuple[int, int] = (8, 6)) -> Tuple["Figure", Axes3D]:
    """Create a 3D matplotlib figure with scientific styling."""
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
    """Plot time series data on the given axes."""
    ax.plot(time_data, data)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    plt.tight_layout()


def plot_from_csv() -> None:
    """Generate plots from saved CSV data."""
    configure_plot()
    agent_types, agents_state_data, target_state_data = get_simulation_data()
    nn_agent_types, agents_nn_data = get_nn_data()

    time_vals: pd.Series = agents_state_data[0]["Time"]

    # Plot tracking error
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

    # Plot 3D trajectories
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
        label="Target",
        linestyle="-",
        color="red",
    )
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.legend(loc="best", fontsize=14, frameon=True, edgecolor="black")
    plt.tight_layout()

    # Plot neural network output
    fig, ax = create_plot_with_config()
    for i, ad in enumerate(agents_state_data):
        nn_output: pd.Series = ad["Neural Network Output Norm"]
        ax.plot(time_vals, nn_output, label=f"{agent_types[i].title()}")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Neural Network Output Norm", fontsize=16)
    ax.legend(loc="best", fontsize=14, frameon=True, edgecolor="black")
    plt.tight_layout()

    # Plot control output
    fig, ax = create_plot_with_config()
    for i, ad in enumerate(agents_state_data):
        control_output: pd.Series = ad["Control Output Norm"]
        ax.plot(time_vals, control_output, label=f"{agent_types[i].title()}")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Control Output Norm", fontsize=16)
    ax.legend(loc="best", fontsize=14, frameon=True, edgecolor="black")
    plt.tight_layout()

    plt.show()


def animate() -> FuncAnimation:
    """Create an animated 3D plot of the simulation."""
    configure_plot()
    agent_types, agents_state_data, target_state_data = get_simulation_data()

    fig, ax = create_3d_plot_with_config()
    trail_length: int = 100

    # Initialize lines and points
    agent_lines: List[Any] = []
    agent_points: List[Any] = []

    for i, _data in enumerate(agents_state_data):
        (line,) = ax.plot([], [], [], label=agent_types[i].title(), linestyle="--", color="blue")
        agent_lines.append(line)
        (point,) = ax.plot([], [], [], "o", color="blue", markersize=6)
        agent_points.append(point)

    (target_line,) = ax.plot([], [], [], label="Target", linestyle="-", color="red")
    (target_point,) = ax.plot([], [], [], "o", color="red", markersize=8)

    # Set up the axes
    all_x = np.concatenate([data["Position X"] for data in agents_state_data] + [target_state_data["Position X"]])
    all_y = np.concatenate([data["Position Y"] for data in agents_state_data] + [target_state_data["Position Y"]])
    all_z = np.concatenate([data["Position Z"] for data in agents_state_data] + [target_state_data["Position Z"]])

    ax.set_xlim(all_x.min(), all_x.max())
    ax.set_ylim(all_y.min(), all_y.max())
    ax.set_zlim(all_z.min(), all_z.max())
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.legend(loc="best", fontsize=14, frameon=True, edgecolor="black")

    # Add time text
    time_text = ax.text2D(0.02, 0.98, "", transform=ax.transAxes, fontsize=12, verticalalignment="top")

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
    """Generate all plots and visualizations from saved data."""
    close_all_files()
    plot_from_csv()