"""Command-line interface for the Online Adaptive Deep Residual Neural Network."""

import argparse
import sys
from typing import Any, Dict, List

import numpy as np

from ..core import dynamics
from ..core.entity import Agent, Target
from ..io.config import load_config
from ..io.data_manager import close_all_files, save_nn_to_csv, save_state_to_csv
from ..visualization.plotter import results


def run_simulation(config: Dict[str, Any]) -> None:
    """Run the main simulation loop.
    
    Args:
        config: Configuration dictionary with simulation parameters
    """
    final_time: float = config["final_time"]
    time_step_delta: float = config["time_step_delta"]
    time_steps: int = int(final_time / time_step_delta)
    num_states: int = config["num_states"]
    np.random.seed(config["seed"])

    if "target_initial_conditions" in config:
        target_position: np.ndarray = np.array(config["target_initial_conditions"])
    else:
        target_position = np.array(
            dynamics.get_initial_conditions(config.get("dynamics_type", "trophic_dynamics"))
        )

    target: Target = Target(target_position, time_steps, config)

    agent_position: np.ndarray = np.zeros(num_states)
    agent: Agent = Agent(agent_position, time_steps, config, target, config["ID"])
    agents: List[Agent] = [agent]

    for step in range(1, time_steps):
        for agent in agents:
            agent.compute_control_output(step)
        for agent in agents:
            agent.update_dynamics(step)
        target.update_dynamics(step)

        time_sim: float = step * time_step_delta
        save_state_to_csv(step, time_sim, agents, target)
        save_nn_to_csv(step, time_sim, agents)

        print(f"Progress: {step / time_steps * 100:6.2f}%", end="\r", flush=True)

    print("\nSimulation completed.")
    close_all_files()
    results()


def main() -> None:
    """Main entry point for the CLI."""
    parser = argparse.ArgumentParser(
        description="Online Adaptive Deep Residual Neural Network Simulation"
    )
    parser.add_argument(
        "--config",
        "-c",
        default="config.json",
        help="Path to the configuration file (default: config.json)",
    )
    parser.add_argument(
        "--plot-only",
        "-p",
        action="store_true",
        help="Only generate plots from existing data (skip simulation)",
    )
    
    args = parser.parse_args()
    
    if args.plot_only:
        print("Generating plots from existing data...")
        results()
        return
    
    try:
        config = load_config(args.config)
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error loading configuration: {e}", file=sys.stderr)
        sys.exit(1)
    
    print(f"Starting simulation with config: {args.config}")
    run_simulation(config)


if __name__ == "__main__":
    main()