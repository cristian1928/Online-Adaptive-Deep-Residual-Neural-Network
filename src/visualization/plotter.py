import os
from typing import Any, Dict, List, Tuple, cast
from matplotlib.colors import ListedColormap

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scienceplots  # type: ignore

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
        'lines.linewidth': 1.5,
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

def get_color_map(agent_types: List[str]) -> Dict[str, Tuple[float, ...]]:
    """Create a chronological color map by pulling from a standard, discrete color list."""
    cmap = plt.get_cmap('tab20')
    listed_cmap = cast(ListedColormap, cmap)
    standard_colors = cast(List[Tuple[float, ...]], listed_cmap.colors)
    color_map: Dict[str, Tuple[float, ...]] = {}
    for i, agent_type in enumerate(agent_types):
        color_map[agent_type] = standard_colors[i % len(standard_colors)]
    return color_map

def plot_from_csv() -> None:
    """Generate all plots from CSV simulation data."""
    configure_plot()
    agent_types, agents_state_data, target_state_data = get_simulation_data()
    nn_agent_types, agents_nn_data = get_nn_data()

    color_map = get_color_map(agent_types)
    time_vals = agents_state_data[0]['Time']

    # ─── Tracking Error Norm ───
    fig_te, ax_te = plt.subplots(figsize=(8, 6))
    plot_data = []
    for i, ad in enumerate(agents_state_data):
        te = ad['Tracking Error Norm']
        rms = np.sqrt(np.mean(te**2))
        plot_data.append((agent_types[i], te, rms))
    plot_data.sort(key=lambda x: x[2], reverse=True)
    for agent_type, te, rms in plot_data:
        ax_te.plot(time_vals, te, label=f'{agent_type.title()}: RMS {rms:.4f} m', color=color_map[agent_type], linestyle='solid')
    ax_te.set_xlabel('Time (s)')
    ax_te.set_ylabel('Tracking Error Norm (m)')
    ax_te.legend(loc='best', fontsize=12, frameon=True, edgecolor='black')
    plt.tight_layout()

    # ─── Spatial Trajectories over Time ───
    fig_traj = plt.figure(figsize=(8, 6))
    ax_traj = fig_traj.add_subplot(111, projection='3d')
    for i, pos in enumerate(agents_state_data):
        x_vals = cast(Any, pos['Position X'].values)
        y_vals = cast(Any, pos['Position Y'].values)
        z_vals = cast(Any, pos['Position Z'].values)
        ax_traj.plot(x_vals, y_vals, z_vals, label=agent_types[i].title(), linestyle='solid', color=color_map[agent_types[i]])
    target_x = cast(Any, target_state_data['Position X'])
    target_y = cast(Any, target_state_data['Position Y'])
    target_z = cast(Any, target_state_data['Position Z'])
    ax_traj.plot(target_x, target_y, target_z, label='Target Trajectory', linestyle='dotted', color='black', linewidth=2.0)
    ax_traj.set_xlabel('X Position (m)')
    ax_traj.set_ylabel('Y Position (m)')
    ax_traj.set_zlabel('Z Position (m)')  # type: ignore[attr-defined]
    ax_traj.legend(loc='best', fontsize=12, frameon=True, edgecolor='black')
    ax_traj.set_box_aspect((1, 1, 1))  # type: ignore[arg-type]
    plt.tight_layout()

    # ─── Neural Network Weights (One Plot Per ID) ───
    for i, nn in enumerate(agents_nn_data):
        fig_nn_w, ax_nn_w = plt.subplots(figsize=(8, 6))
        time_nn = nn['Time']
        weight_cols = [c for c in nn.columns if c.startswith('Weight_')]
        for col in weight_cols:
            ax_nn_w.plot(time_nn, nn[col], linestyle='solid')
        ax_nn_w.set_title(f'Neural Network Weights for {nn_agent_types[i].title()}')
        ax_nn_w.set_xlabel('Time (s)')
        ax_nn_w.set_ylabel('Weight Value')
        plt.tight_layout()

    # ─── Overlaid NN-related Plots ───
    fig_fae, ax_fae = plt.subplots(figsize=(8, 6))
    fig_lrs, ax_lrs = plt.subplots(figsize=(8, 6))
    fig_nno, ax_nno = plt.subplots(figsize=(8, 6))

    for i, nn in enumerate(agents_nn_data):
        agent_id = nn_agent_types[i]
        color = color_map.get(agent_id, 'k')
        time_nn = nn['Time']
        label = agent_id.title()

        ax_fae.plot(time_nn, nn['Function Approximation Error Norm'], label=label, color=color, linestyle='solid')
        ax_lrs.plot(time_nn, nn['Learning Rate Spectral Norm'], label=label, color=color, linestyle='solid')
        ax_nno.plot(time_nn, nn['Neural Network Output'], label=label, color=color, linestyle='solid')

    ax_fae.set_xlabel('Time (s)')
    ax_fae.set_ylabel('Function Approximation Error Norm')
    ax_fae.legend(loc='best', fontsize=12, frameon=True, edgecolor='black')
    fig_fae.tight_layout()

    ax_lrs.set_xlabel('Time (s)')
    ax_lrs.set_ylabel('Learning Rate Spectral Norm')
    ax_lrs.legend(loc='best', fontsize=12, frameon=True, edgecolor='black')
    fig_lrs.tight_layout()

    ax_nno.set_xlabel('Time (s)')
    ax_nno.set_ylabel('Neural Network Output $(m/s^2)$')
    ax_nno.legend(loc='best', fontsize=12, frameon=True, edgecolor='black')
    fig_nno.tight_layout()

    plt.show()

def results() -> None:
    """Generate all results plots and visualizations."""
    plot_from_csv()

if __name__ == "__main__":
    results()