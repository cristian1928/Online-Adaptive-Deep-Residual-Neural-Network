import numpy as np
import json
from entity import Agent, Target
from data_manager import save_nn_to_csv, save_state_to_csv, results, close_all_files
import dynamics

def run_simulation(config):
    # Setup simulation parameters
    final_time = config['final_time']
    time_step_delta = config['time_step_delta']
    time_steps = int(final_time / time_step_delta)
    num_states = config['num_states']
    np.random.seed(config['seed'])

    if 'target_initial_conditions' in config: target_position = np.array(config['target_initial_conditions'])
    else: target_position = np.array(dynamics.get_initial_conditions(config.get('dynamics_type', 'trophic_dynamics')))
    
    target = Target(target_position, time_steps, config)

    # Initialize agent
    agent_position = np.zeros(num_states)
    agent = Agent(agent_position, time_steps, config, target, config['ID'])
    agents = [agent]

    # Main simulation loop
    for step in range(1, time_steps):
        # Update all agents
        for agent in agents: agent.compute_control_output(step)
        for agent in agents: agent.update_dynamics(step)
        target.update_dynamics(step)

        # Save data
        time_sim = step * time_step_delta
        save_state_to_csv(step, time_sim, agents, target)
        save_nn_to_csv(step, time_sim, agents)

        # Progress display
        print(f'Progress: {step / time_steps * 100:6.2f}%', end='\r', flush=True)

    print("\nSimulation completed.")
    close_all_files()

def run_simulation_with_results(config):
    """Run simulation and generate plots/animations."""
    run_simulation(config)
    results()

if __name__ == "__main__":
    with open('config.json', 'r') as config_file: config = json.load(config_file)
    run_simulation_with_results(config)