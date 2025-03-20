import numpy as np
from entity import Agent, Target
from data_manager import save_nn_to_csv, save_state_to_csv, results
import json

class Simulation:
    def __init__(self, config):
        self.final_time = config['final_time']
        self.time_step_delta = config['time_step_delta']
        self.time_steps = int(self.final_time / self.time_step_delta)
        self.num_states = config['num_states']
        np.random.seed(0)

        target_position = np.random.uniform(-10, 10, self.num_states)
        target_velocity = np.zeros(self.num_states)
        self.target = Target(target_position, target_velocity, self.time_steps, config)

        agent_position = np.zeros(self.num_states)
        agent_velocity = np.zeros(self.num_states)
        control1 = Agent(agent_position, agent_velocity, self.time_steps, config, self.target, config['ID'])
        self.agents = [control1]

    def run(self):
        for step in range(1, self.time_steps):
            for agent in self.agents: agent.compute_control_output(step)
            for agent in self.agents: agent.update_dynamics(step)
            self.target.update_dynamics(step)
            time = step * self.time_step_delta
            save_state_to_csv(step, time, self.agents, self.target)
            save_nn_to_csv(step, time, self.agents)
            print(f"Progress: {step/self.time_steps*100:.2f}%", end='\r', flush=True)
        print("\nSimulation completed.")
        results()

if __name__ == "__main__":
    with open('config.json', 'r') as config_file: config = json.load(config_file)
    simulation = Simulation(config)
    simulation.run()