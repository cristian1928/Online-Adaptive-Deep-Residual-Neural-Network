import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib as mpl

# Ensure output directory exists
def ensure_directory_exists(directory: str):
    os.makedirs(directory, exist_ok=True)

# Save agent and target states to CSV
def save_state_to_csv(step, time, agents, target):
    ensure_directory_exists('simulation_data')

    for agent in agents:
        tracking_error_norm = np.linalg.norm(agent.tracking_error)

        state_data = pd.DataFrame({
            'Time': [time],
            'Position_X': [agent.positions[0, step - 1]],
            'Position_Y': [agent.positions[1, step - 1]],
            'Position_Z': [agent.positions[2, step - 1]],
            'Tracking_Error_Norm': [tracking_error_norm],
        })

        # Construct the file path using the new ID
        state_file_path = f'simulation_data/{agent.agent_type}_state_data.csv'

        if step == 1:
            # If file exists, remove it so a fresh CSV is created
            if os.path.exists(state_file_path):
                os.remove(state_file_path)
            # Write header on the first write
            state_data.to_csv(state_file_path, index=False, header=True)
        else:
            # Append subsequent data
            state_data.to_csv(state_file_path, mode='a', header=False, index=False)

    target_state_data = pd.DataFrame({
        'Time': [time],
        'Position_X': [target.positions[0, step - 1]],
        'Position_Y': [target.positions[1, step - 1]],
        'Position_Z': [target.positions[2, step - 1]],
    })

    target_file_path = 'simulation_data/target_state_data.csv'
    if step == 1:
        if os.path.exists(target_file_path):
            os.remove(target_file_path)
        target_state_data.to_csv(target_file_path, index=False, header=True)
    else:
        target_state_data.to_csv(target_file_path, mode='a', header=False, index=False)

# Save neural network data to CSV
def save_nn_to_csv(step, time, agents):
    ensure_directory_exists('simulation_data')

    for agent in agents:
        # Convert each weight to a float
        # If 'weights' is a list/array of shape (N,), flatten them to pure floats:
        float_weights = []
        for w in agent.neural_network.weights:
            # If w is, for instance, np.array([1.23]) or [1.23], unbox it:
            if isinstance(w, (list, np.ndarray)) and len(w) == 1:
                float_weights.append(float(w[0]))
            else:
                float_weights.append(float(w))

        # Build DataFrame with purely numeric columns
        nn_data = pd.DataFrame({
            'Time': [time],
            **{f'Weight_{j + 1}': [float_weights[j]] for j in range(len(float_weights))},
        })

        nn_file_path = f'simulation_data/{agent.agent_type}_nn_data.csv'

        # Write headers if step == 1, else append
        if step == 1:
            if os.path.exists(nn_file_path):
                os.remove(nn_file_path)
            nn_data.to_csv(nn_file_path, index=False, header=True)
        else:
            nn_data.to_csv(nn_file_path, mode='a', header=False, index=False)

# Constants for IEEE standard plotting
IEEE_FIGSIZE = (10, 8)
IEEE_FONTSIZE = 10
IEEE_LINEWIDTH = 1.5
IEEE_GRID_STYLE = {'linestyle': '--', 'linewidth': 0.5, 'color': 'gray'}

mpl.rcParams['savefig.format'] = 'eps'

# Plotting functions adhering to IEEE standards
def configure_plot():
    plt.rcParams['text.usetex'] = True
    plt.rcParams.update({
        'font.size': IEEE_FONTSIZE,
        'lines.linewidth': IEEE_LINEWIDTH,
    })

def plot_from_csv():
    configure_plot()

    csv_state_files = [f for f in os.listdir('simulation_data') if f.endswith('_state_data.csv') and not f.startswith('target')]
    csv_state_files.sort()
    agent_types = [f.replace('_state_data.csv', '') for f in csv_state_files]
    agents_state_data = [pd.read_csv(os.path.join('simulation_data', f)) for f in csv_state_files]

    csv_nn_files = [f for f in os.listdir('simulation_data') if f.endswith('_nn_data.csv')]
    csv_nn_files.sort()
    nn_state_data = [pd.read_csv(os.path.join('simulation_data', f)) for f in csv_nn_files]

    # Read data from CSV files
    agents_state_data = [pd.read_csv(f'simulation_data/{atype}_state_data.csv') for atype in agent_types]
    target_state_data = pd.read_csv('simulation_data/target_state_data.csv')
    time_array = target_state_data['Time']

    # Plot tracking error over time
    plt.figure(figsize=IEEE_FIGSIZE)
    for i, agent_state_data in enumerate(agents_state_data):
        tracking_error_norm = agent_state_data['Tracking_Error_Norm']
        rms_tracking_error = np.sqrt(np.mean(tracking_error_norm**2))
        plt.plot(time_array, tracking_error_norm, label=f'{agent_types[i].title()}: RMS {rms_tracking_error:.4f} m')
    plt.xlabel('Time (s)')
    plt.ylabel('Tracking Error Norm $(m)$')
    plt.legend(loc='best', fontsize=IEEE_FONTSIZE, frameon=True)
    plt.grid(**IEEE_GRID_STYLE)
    plt.tight_layout()

    # Plot neural network weights over time
    plt.figure(figsize=IEEE_FIGSIZE)
    for i, nn_data in enumerate(nn_state_data):
        time_array = nn_data['Time']
        weight_columns = [col for col in nn_data.columns if col.startswith('Weight_')]
        for wcol in weight_columns: plt.plot(time_array, nn_data[wcol], label=f'{agent_types[i].title()} - {wcol}')
    plt.xlabel('Time (s)')
    plt.ylabel('DNN Weight Value')
    plt.grid(**IEEE_GRID_STYLE)

    # Plot function approximation error over time
    #plt.figure(figsize=IEEE_FIGSIZE)
    #for i, agent_state_data in enumerate(agents_state_data):
    #    time_array = agent_state_data['Time']
    #    error_columns = [col for col in agent_state_data.columns if col.startswith('Function_')]
    #    for ecol in error_columns: plt.plot(time_array, agent_state_data[ecol], label=f'Agent {i+1} - {ecol}')
    #plt.xlabel('Time (s)')
    #plt.ylabel('Function Approximation Error $(m/s)$')
    #plt.grid(**IEEE_GRID_STYLE)

    # Plot positions in 3D space (including target)
    fig = plt.figure(figsize=IEEE_FIGSIZE)
    ax = fig.add_subplot(111, projection='3d')
    for i, agent_state_data in enumerate(agents_state_data):
       ax.plot(agent_state_data['Position_X'],  agent_state_data['Position_Y'], agent_state_data['Position_Z'], label=f'{agent_types[i].title()}')
    ax.plot(target_state_data['Position_X'], target_state_data['Position_Y'], target_state_data['Position_Z'], label='Target', linestyle='--')
    ax.set_xlabel('X Position $(m)$')
    ax.set_ylabel('Y Position $(m)$')
    ax.set_zlabel('Z Position $(m)$')
    ax.legend()
    plt.grid(**IEEE_GRID_STYLE)
    plt.tight_layout()

    # Show all plots
    plt.show()

def results():
    plot_from_csv()