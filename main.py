import numpy as np
import numpy.typing as npt
import json
from typing import Dict, Any
from src.core.entity import Agent, Target
from src.io.data_manager import save_nn_to_csv, save_state_to_csv, close_all_files
from src.visualization.plotter import results
from src.simulation import dynamics

def run_simulation(config: Dict[str, Any]) -> None:
    # Setup simulation parameters
    final_time: float = config['final_time']
    time_step_delta: float = config['time_step_delta']
    time_steps: int = int(final_time / time_step_delta)
    num_states: int = config['num_states']
    np.random.seed(config['seed'])

    if 'target_initial_conditions' in config: 
        target_position: npt.NDArray[np.floating[Any]] = np.array(config['target_initial_conditions'])
    else: 
        target_position = np.array(dynamics.get_initial_conditions(config.get('dynamics_type', 'trophic_dynamics')))
    
    target: Target = Target(target_position, time_steps, config)

    # Initialize agent
    agent_position: npt.NDArray[np.floating[Any]] = np.zeros(num_states)
    agent: Agent = Agent(agent_position, time_steps, config, target, config['ID'])
    agents = [agent]

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

def run_simulation_with_results(config: Dict[str, Any]) -> None:
    """Run simulation and generate plots/animations."""
    run_simulation(config)
    results()

if __name__ == "__main__":
    with open('config.json', 'r') as config_file: 
        config: Dict[str, Any] = json.load(config_file)
    run_simulation_with_results(config)