import numpy as np
import time
import json
from entity import Agent, Target
from data_manager import save_nn_to_csv, save_state_to_csv, results, close_all_files

class Simulation:
    def __init__(self, config):
        self.final_time = config['final_time']
        self.time_step_delta = config['time_step_delta']
        self.time_steps = int(self.final_time / self.time_step_delta)
        self.num_states = config['num_states']
        np.random.seed(config['seed'])

        # Initialize target
        target_position = np.array(config['target_initial_conditions'])
        self.target = Target(target_position, self.time_steps, config)

        # Initialize agent
        agent_position = np.zeros(self.num_states)
        agent = Agent(agent_position, self.time_steps, config, self.target, config['ID'])
        self.agents = [agent]

        # Performance tracking
        self.total_loop_time, self.completed_steps = 0.0, 0
        self._sim_start_time = time.time()

    def _fmt(self, seconds): m, s = divmod(seconds, 60); h, m = divmod(m, 60); return f'{int(h):02d}:{int(m):02d}:{s:05.2f}'

    def _display_progress(self, step, loop_start_time):
        self.total_loop_time += time.time() - loop_start_time
        self.completed_steps += 1
        avg_loop_time = self.total_loop_time / self.completed_steps
        progress = step / self.time_steps * 100
        elapsed = self._fmt(time.time() - self._sim_start_time)
        print(f'Progress: {progress:6.2f}% | Elapsed: {elapsed} | Avg loop: {avg_loop_time:8.6f}s', end='\r', flush=True)

    def run(self):
        for step in range(1, self.time_steps):
            loop_start = time.time()

            # Update all agents
            for agent in self.agents: agent.compute_control_output(step)
            for agent in self.agents: agent.update_dynamics(step)
            self.target.update_dynamics(step)
            
            # Save data
            time_sim = step * self.time_step_delta
            save_state_to_csv(step, time_sim, self.agents, self.target)
            save_nn_to_csv(step, time_sim, self.agents)
            
            # Performance tracking and progress display
            self._display_progress(step, loop_start)
            
        print("\nSimulation completed.")
        close_all_files()
        results()

if __name__ == "__main__":
    with open('config.json', 'r') as config_file: config = json.load(config_file)
    simulation = Simulation(config)
    simulation.run()