import os
from typing import Any, List, Tuple, cast

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.animation import FuncAnimation
from matplotlib.lines import Line2D

# Constants for data access
DATA_DIR = 'simulation_data'
STATE_DATA_SUFFIX = '_state_data.csv'
NN_DATA_SUFFIX = '_nn_data.csv'
TARGET_FILE = f'{DATA_DIR}/target_state_data.csv'

def configure_plot() -> None:
    """Configure matplotlib for IEEE standard plotting."""
    plt.style.use(['science', 'ieee'])
    plt.rcParams['figure.dpi'] = 100
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["axes.labelsize"] = 14
    plt.rcParams["axes.titlesize"] = 16
    plt.rcParams["xtick.labelsize"] = 12
    plt.rcParams["ytick.labelsize"] = 12
    plt.rcParams.update({
        'lines.linewidth': 1.0,
        'axes.linewidth': 0.5,
        'legend.frameon': True,
        'legend.edgecolor': 'black',
    })

def get_simulation_data() -> Tuple[List[str], List[pd.DataFrame], pd.DataFrame]:
    """Load simulation state data from CSV files."""
    csv_state_files = [f for f in os.listdir(DATA_DIR) if f.endswith(STATE_DATA_SUFFIX) and not f.startswith('target')]
    csv_state_files.sort()
    agent_types = [f.replace(STATE_DATA_SUFFIX, '') for f in csv_state_files]
    agents_state_data = [pd.read_csv(os.path.join(DATA_DIR, f)) for f in csv_state_files]
    target_state_data = pd.read_csv(TARGET_FILE)
    return agent_types, agents_state_data, target_state_data

def get_nn_data() -> Tuple[List[str], List[pd.DataFrame]]:
    """Load neural network data from CSV files."""
    csv_nn_files = [f for f in os.listdir(DATA_DIR) if f.endswith(NN_DATA_SUFFIX)]
    csv_nn_files.sort()
    agent_types = [f.replace(NN_DATA_SUFFIX, '') for f in csv_nn_files]
    agents_nn_data = [pd.read_csv(os.path.join(DATA_DIR, f)) for f in csv_nn_files]
    return agent_types, agents_nn_data

def plot_from_csv() -> None:
    """Generate all plots from CSV simulation data."""
    configure_plot()
    agent_types, agents_state_data, target_state_data = get_simulation_data()
    nn_agent_types, agents_nn_data = get_nn_data()

    # ─── Time ───
    time_vals = agents_state_data[0]['Time']

    # ─── Tracking Error Norm ───
    fig, ax = plt.subplots(figsize=(8, 6))
    plot_data = []
    for i, ad in enumerate(agents_state_data):
        te = ad['Tracking Error Norm']
        rms = np.sqrt(np.mean(te**2))
        plot_data.append((i, agent_types[i], te, rms))
    plot_data.sort(key=lambda x: x[3], reverse=True)
    for i, agent_type, te, rms in plot_data:
        ax.plot(time_vals, te, label=f'{agent_type.title()}: RMS {rms:.4f} m')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Tracking Error Norm (m)', fontsize=16)
    ax.legend(loc='best', fontsize=14, frameon=True, edgecolor='black')
    plt.tight_layout()

    # ─── Spatial Trajectories over Time ───
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    for i, pos in enumerate(agents_state_data):
        x_vals = cast(Any, pos['Position X'].values)
        y_vals = cast(Any, pos['Position Y'].values) 
        z_vals = cast(Any, pos['Position Z'].values)
        ax.plot(x_vals, y_vals, z_vals, label=agent_types[i].title(), linestyle='--', color='blue')
    
    target_x = cast(Any, target_state_data['Position X'])
    target_y = cast(Any, target_state_data['Position Y'])
    target_z = cast(Any, target_state_data['Position Z'])
    ax.plot(target_x, target_y, target_z, label='Target Trajectory', linestyle='--', color='black')
    ax.set_xlabel('X Position (m)')
    ax.set_ylabel('Y Position (m)')
    ax.set_zlabel('Z Position (m)')  # type: ignore[attr-defined]
    ax.legend(loc='best', fontsize=12, frameon=True, edgecolor='black')
    ax.set_box_aspect((1, 1, 1))  # type: ignore[arg-type]
    plt.tight_layout()

    # ─── Neural Network Weights ───
    for i, nn in enumerate(agents_nn_data):
        fig, ax = plt.subplots(figsize=(8, 6))
        time_nn = nn['Time']
        weight_cols = [c for c in nn.columns if c.startswith('Weight_')]
        for col in weight_cols: 
            ax.plot(time_nn, nn[col])
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Weight Value')
        plt.tight_layout()

    # ─── Function Approximation Error Norm ───
    for i, nn in enumerate(agents_nn_data):
        fig, ax = plt.subplots(figsize=(8, 6))
        time_nn = nn['Time']
        ax.plot(time_nn, nn['Function Approximation Error Norm'])
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Function Approximation Error Norm')
        plt.tight_layout()

    # ─── Learning Rate Spectral Norm ───
    for i, nn in enumerate(agents_nn_data):
        fig, ax = plt.subplots(figsize=(8, 6))
        time_nn = nn['Time']
        ax.plot(time_nn, nn['Learning Rate Spectral Norm'])
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Learning Rate Spectral Norm')
        plt.tight_layout()

    # ─── Neural Network Output ───
    for i, nn in enumerate(agents_nn_data):
        fig, ax = plt.subplots(figsize=(8, 6))
        time_nn = nn['Time']
        ax.plot(time_nn, nn['Neural Network Output'])
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Neural Network Output $(m/s^2)$')
        plt.tight_layout()

    plt.show() 

def animate() -> FuncAnimation:
    """Create animated visualization of simulation trajectories."""
    plt.style.use('default')
    agent_types, agents_state_data, target_state_data = get_simulation_data()

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel('X Position (m)')
    ax.set_ylabel('Y Position (m)')
    ax.set_zlabel('Z Position (m)')  # type: ignore[attr-defined]

    # Find data bounds for consistent scaling
    position_data = agents_state_data + [target_state_data]
    x_min, x_max = min(data['Position X'].min() for data in position_data), max(data['Position X'].max() for data in position_data)
    y_min, y_max = min(data['Position Y'].min() for data in position_data), max(data['Position Y'].max() for data in position_data)
    z_min, z_max = min(data['Position Z'].min() for data in position_data), max(data['Position Z'].max() for data in position_data)

    margin = 0.1
    x_range, y_range, z_range = x_max - x_min, y_max - y_min, z_max - z_min
    ax.set_xlim(x_min - margin * x_range, x_max + margin * x_range)
    ax.set_ylim(y_min - margin * y_range, y_max + margin * y_range)
    ax.set_zlim(z_min - margin * z_range, z_max + margin * z_range)  # type: ignore[attr-defined]

    # Initialize lines and points
    agent_lines: List[Line2D] = []
    agent_points: List[Line2D] = []
    for i, agent_data in enumerate(agents_state_data):
        line, = ax.plot([], [], [], '-', label=f'{agent_types[i].title()}')
        point, = ax.plot([], [], [], 'o', markersize=6)
        agent_lines.append(line)
        agent_points.append(point)

    target_line, = ax.plot([], [], [], '--', label='Target')
    target_point, = ax.plot([], [], [], 'o', markersize=6, color='red')

    legend = ax.legend(loc='upper right', prop={'size': 7})
    legend.get_frame().set_linewidth(0.5)
    time_text = ax.text2D(0.02, 0.95, '', transform=ax.transAxes)  # type: ignore[attr-defined]

    trail_length = 3500

    def update(frame: int) -> List[Any]:
        current_time = target_state_data['Time'][frame]
        time_text.set_text(f'Time: {current_time:.2f} s')
    
        start_idx = max(0, frame - trail_length)

        for i, data in enumerate(agents_state_data):
            agent_lines[i].set_data(data['Position X'][start_idx:frame+1], data['Position Y'][start_idx:frame+1])
            agent_lines[i].set_3d_properties(data['Position Z'][start_idx:frame+1])  # type: ignore[attr-defined]
            agent_points[i].set_data([data['Position X'][frame]], [data['Position Y'][frame]])
            agent_points[i].set_3d_properties([data['Position Z'][frame]])  # type: ignore[attr-defined]

        target_line.set_data(target_state_data['Position X'][start_idx:frame+1], target_state_data['Position Y'][start_idx:frame+1])
        target_line.set_3d_properties(target_state_data['Position Z'][start_idx:frame+1])  # type: ignore[attr-defined]
        target_point.set_data([target_state_data['Position X'][frame]], [target_state_data['Position Y'][frame]])
        target_point.set_3d_properties([target_state_data['Position Z'][frame]])  # type: ignore[attr-defined]

        return agent_lines + agent_points + [target_line, target_point, time_text]

    num_frames = len(target_state_data)

    plt.tight_layout()
    plt.show()
    return FuncAnimation(fig, update, frames=range(0, num_frames, max(1, num_frames//200)), blit=False, interval=75)

def results() -> None:
    """Generate all results plots and visualizations."""
    plot_from_csv()
    animate()