from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from src.core.entity import Agent, Target
from src.io.data_manager import close_all_files, save_nn_to_csv, save_state_to_csv
from src.simulation import dynamics
from src.visualization.plotter import results


def run_simulation_from_configs(configs: list[dict[str, Any]]) -> None:
    base_config = configs[0]
    
    # Setup simulation parameters
    final_time: float = base_config['final_time']
    time_step_delta: float = base_config['time_step_delta']
    time_steps: int = int(final_time / time_step_delta)
    num_states: int = base_config['num_states']
    np.random.seed(base_config['seed'])
    dynamics_type = base_config['dynamics_type']
    target_position = np.array(dynamics.get_initial_conditions(dynamics_type))
    target: Target = Target(target_position, time_steps, base_config)

    # Initialize agents from all configurations
    agents: list[Agent] = []
    for config in configs:
        agent_position: NDArray[np.float64] = np.zeros(num_states)
        agent: Agent = Agent(agent_position, time_steps, config, target, config['ID'])
        agents.append(agent)

    # Main simulation loop
    for step in range(1, time_steps):
        # Update all agents
        for agent in agents: agent.compute_control_output(step)
        for agent in agents: agent.update_dynamics(step)
        target.update_dynamics(step)

        # Save data
        time_sim: float = step * time_step_delta
        save_state_to_csv(step, time_sim, agents, target)
        save_nn_to_csv(step, time_sim, agents)

        # Progress display
        print(f'Progress: {step / time_steps * 100:6.2f}%', end='\r', flush=True)

    print("\nSimulation completed.")
    close_all_files()

def run_simulation(config: dict[str, Any]) -> None:
    run_simulation_from_configs([config])

def run_simulation_with_results(config: dict[str, Any]) -> None:
    run_simulation(config)
    results()

def load_configurations() -> list[dict[str, Any]]:
    config_dir = Path("configurations")
    config_files = list(config_dir.glob("*.json"))
    baseline_file = config_dir / "config_common.json"
    baseline_config = {}
    if baseline_file.exists():
        with open(baseline_file, 'r') as f: baseline_config = json.load(f)
    configs = []
    for config_file in sorted(config_files):
        if config_file.name == "config_common.json": continue    
        with open(config_file, 'r') as f:
            config = json.load(f)
            merged_config = {**baseline_config, **config}
            configs.append(merged_config)
    return configs

def run_batch_simulation_with_results() -> None:
    configs = load_configurations()
    run_simulation_from_configs(configs)
    results()

if __name__ == "__main__":
    run_batch_simulation_with_results()