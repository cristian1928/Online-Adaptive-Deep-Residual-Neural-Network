# Online Adaptive Deep Residual Neural Network

A Python-based implementation of an online adaptive deep neural network with residual connections, designed for modeling and controlling nonlinear dynamical systems. The system simulates autonomous dynamics and performs real-time weight adaptation during operation.

## Quickstart Guide

### Prerequisites

- Python 3.14 or higher
- All dependencies listed in `requirements.txt`

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/cristian1928/Online-Adaptive-Deep-Residual-Neural-Network.git
   cd Online-Adaptive-Deep-Residual-Neural-Network
   ```

2. Create and activate a virtual environment:
   ```bash
   # Create the environment
   python3 -m venv venv

   # Activate on macOS/Linux
   source venv/bin/activate
   ```

3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Running the Simulation

Execute the main simulation script:

```bash
python3 main.py
```

This script will:

- Load configuration files from the `configurations/` directory
- Instantiate agents using the specified parameters
- Run the simulation with online adaptive learning
- Save output data to the `simulation_data/` directory
- Generate plots for result analysis following IEEE formatting guidelines

### Visualizing Results

After the simulation, run the plotting script to generate and display all result plots:

```bash
python3 src/visualization/plotter.py
```

## Overview

### Purpose

This framework applies online adaptive control via deep residual neural networks for nonlinear dynamical systems. It supports real-time target tracking with continuous adaptation of network weights based on tracking error.

### Architecture

- **Neural Network Core**: Deep residual architecture with online weight updates
- **Dynamical Systems**: Includes pre-built models for spacecraft attitude control, chaotic systems, and ecological dynamics
- **Simulation Engine**: Real-time numerical integration and state evolution
- **Data Management**: Buffered CSV logging for efficient storage
- **Visualization**: IEEE-style plotting and performance metrics

### Supported Systems

- **Attitude MRP**: Spacecraft control using Modified Rodrigues Parameters
- **Chua Circuit**: Chaotic dynamics modeled by a double-scroll circuit
- **Trophic Dynamics**: A three-tier ecological food chain
- **Custom Systems**: User-defined models supported via modular interface

### Configuration

Simulations are configured using JSON files located in the `configurations/` directory. Each file defines the parameters for one agent, including network architecture and learning settings. Multiple configuration files enable batch simulations.

## Code Quality and Testing

To maintain code integrity, the following checks are recommended before committing changes:

### Static Type Checking

Run this after modifying type hints or function signatures:

```bash
mypy --strict
```

### Unit and Integration Tests

Run the complete test suite:

```bash
python -m pytest -v tests/
```

## License

This project is licensed under the GNU Affero General Public License v3.0. See the [LICENSE](LICENSE) file for more information.

## Citation

If you use this work in your research, please cite:

```
@article{Nino.Patil.ea2025,
  author  = {Cristian F. Nino and Omkar Sudhir Patil and Marla R. Eisman and Warren E. Dixon},
  title   = {Online ResNet-Based Adaptive Control for Nonlinear Target Tracking},
  year    = {2025},
  journal = {IEEE Control Systems Letters},
  volume  = {9},
  pages   = {907-912},
  doi     = {10.1109/LCSYS.2025.3576652}
}
```

## Contact

For questions or technical support, contact:

- **Cristian Nino**
- **Email:** cristian1928@ufl.edu
- **GitHub:** [@cristian1928](https://github.com/cristian1928)

For detailed documentation, see [HELP.md](HELP.md).
