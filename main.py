import numpy as np
import json
from typing import Dict, Any, List
from entity import Agent, Target
from data_manager import save_nn_to_csv, save_state_to_csv, results, close_all_files
import dynamics

def run_simulation(config: Dict[str, Any]) -> None:
    final_time: float = config['final_time']
    time_step_delta: float = config['time_step_delta']
    time_steps: int = int(final_time / time_step_delta)
    num_states: int = config['num_states']
    np.random.seed(config['seed'])

    if 'target_initial_conditions' in config:
        target_position: np.ndarray = np.array(config['target_initial_conditions'])
    else:
        target_position = np.array(dynamics.get_initial_conditions(config.get('dynamics_type', 'trophic_dynamics')))
    
    target: Target = Target(target_position, time_steps, config)

    agent_position: np.ndarray = np.zeros(num_states)
    agent: Agent = Agent(agent_position, time_steps, config, target, config['ID'])
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

        print(f'Progress: {step / time_steps * 100:6.2f}%', end='\r', flush=True)

    print("\nSimulation completed.")
    close_all_files()
    results()

if __name__ == "__main__":
    with open('config.json', 'r') as config_file:
        config: Dict[str, Any] = json.load(config_file)
    run_simulation(config)