# Online Adaptive Deep Residual Neural Network

An implementation of an online adaptive deep neural network with residual connections designed for modeling and controlling dynamical systems. This repository provides a simulation framework where a deep neural network continuously learns and adapts its weights in real time, improving its performance with each simulation step.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Architecture](#architecture)
- [Configuration](#configuration)
- [Usage](#usage)
- [License](#license)
- [Contact](#contact)

## Overview

This project combines online adaptive learning with deep residual network architecture to efficiently approximate the dynamics of complex systems. The framework simulates system behavior, continuously updates the network in an online manner, and provides real-time visualization of both the simulation and the training process. It is ideal for researchers and developers looking to experiment with adaptive control algorithms and residual neural network architectures.

## Features

- **Online Adaptive Learning:** Continuously adjusts network weights based on streaming simulation data.
- **Deep Residual Architecture:** Incorporates residual (shortcut) connections to facilitate training deeper networks and to improve convergence.
- **Customizable Simulation Dynamics:** Simulate system dynamics with adjustable parameters (e.g., time step, final simulation time, number of states).
- **Modular Design:** Separated modules for system dynamics, neural network computations, data management, and result visualization.
- **Configurable Network Parameters:** Easily adjust network depth, layer sizes, activation functions, learning rate, and more via a configuration file.
- **Visualization Tools:** Integrated plotting functions for monitoring simulation outcomes and network performance.

## Architecture

The project is organized into several key modules:

- **System Dynamics (`dynamics.py`):**  
  Implements the equations governing the simulated system. This module updates the state of the system over time based on the current dynamics.

- **Neural Network (`neural_network.py`):**  
  Contains the implementation of the deep residual neural network. The network is designed to learn the mapping from inputs to outputs in an online adaptive manner, using specified activations.

- **Entity (`entity.py`):**  
  Encapsulates an agent that interacts with the system dynamics. The entity uses the adaptive network to predict or control system behavior.

- **Data Manager (`data_manager.py`):**  
  Manages the collection, storage, and handling of simulation data. This module is responsible for ensuring that data is readily available for online learning and evaluation.

- **Plotter (`plotter.py`):**  
  Provides functionality to visualize simulation results and performance metrics, offering real-time feedback on the system’s performance and the network's learning progress.

- **Main Application (`main.py`):**  
  Serves as the entry point of the program. It integrates all modules, sets up the simulation according to the configuration, and initiates the online adaptive learning process.

### Prerequisites

- Python 3.7 or higher
- [NumPy](https://numpy.org/)
- [Matplotlib](https://matplotlib.org/) (for visualization)

## Configuration

The project is driven by a JSON configuration file (`config.json`) that specifies both simulation and network parameters. Below is an overview of the key configuration parameters:

- **Simulation Parameters:**
  - `final_time`: Total duration of the simulation (e.g., 60 seconds).
  - `time_step_delta`: The simulation time step (e.g., 0.01 seconds).
  - `num_states`: Number of states in the system.
  - `num_agents`: Number of agents interacting with the system.

- **Neural Network Architecture:**
  - `num_blocks`: Number of residual blocks to use (set to 0 if not using additional residual layers).
  - `num_layers`: Total number of layers in the network.
  - `num_neurons`: Number of neurons per layer.
  - `num_inputs`: Dimensionality of the input vector.
  - `num_outputs`: Dimensionality of the output vector.
  - `inner_activation`: Activation function for the inner layers. Options include:
    - "tanh": Hyperbolic tangent, outputs between -1 and 1
    - "swish": Self-gated activation (x * sigmoid(x))
    - "identity": No activation, f(x) = x
    - "relu": Rectified Linear Unit, max(0, x)
    - "sigmoid": Logistic function, outputs between 0 and 1
    - "leaky_relu": Modified ReLU with small negative slope (0.01)
  - `output_activation`: Activation function for the output layer (same options as inner_activation).
  - `shortcut_activation`: Activation function used for residual shortcuts (same options as inner_activation).

- **Learning Parameters:**
  - `learning_rate`: The step size used during network training.
  - `weight_bounds`: Maximum allowable magnitude for network weights.
  - `k1`: A scaling factor or gain parameter used in the adaptive algorithm.
  - `forgetting_factor`: Parameter controlling the rate at which past information is “forgotten” during online learning.

You can modify these parameters to experiment with different simulation settings and network architectures.

## Usage

To run the simulation and begin online adaptive learning, execute the main script:

```bash
python main.py
```
This will:
- Initialize the system dynamics and network with the parameters specified in config.json.
- Start the simulation loop where the system state is updated at each time step.
- Adapt the neural network weights in an online manner.
- Log data via the data manager and update visualizations in real time using the plotter.

## License

This project is licensed under the GNU Affero General Public License v3.0. See the [LICENSE](LICENSE) file for details.

## Contact

For questions, suggestions, or further information, please contact:

- **Name:** Cristian Nino
- **Email:** cristian1928@ufl.edu
- **GitHub:** [@cristian1928](https://github.com/cristian1928)
