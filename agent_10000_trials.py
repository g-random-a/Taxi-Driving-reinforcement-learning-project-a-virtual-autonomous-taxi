import random
from environment import Agent, Environment
from planner import RoutePlanner
from simulator import Simulator
import pandas as pd
import re

class LearningAgent(Agent):
    """An agent that learns to drive in the smartcab world."""

    def __init__(self, env):
        super().__init__(env)  # Call the parent class (Agent) constructor
        self.color = 'red'  # Override the default agent color
        self.planner = RoutePlanner(self.env, self)  # Initialize a route planner to get the next waypoint
        self.qs = {}  # Initialize Q-table to store Q-values for state-action pairs
        self.time = 0  # Initialize time step counter
        self.errors = 0  # Initialize error counter
        self.possible_actions = (None, 'left', 'forward', 'right')  # Define possible actions
        self.optimal_val = 0  # Initialize optimal Q-value

    def reset(self, destination=None):
        """
        Reset the agent for a new trial.
        
        Parameters:
        destination (tuple): The destination to route to.
        """
        self.planner.route_to(destination)  # Route to the specified destination
        # TODO: Prepare for a new trip; reset any variables here, if required
        # Currently, no additional variables to reset

    def best_action(self, state):
        """
        Returns the best action (the one with the maximum Q-value)
        or one of the best actions, given a state.
        
        Parameters:
        state (tuple): The current state of the agent.
        
        Returns:
        action (str): The best action for the given state.
        """
        # Get all possible Q-values for the state
        all_qs = {action: self.qs.get((state, action), 0) for action in self.possible_actions}
        
        # Pick the actions that yield the largest Q-value for the state
        optimal_actions = [action for action in self.possible_actions if all_qs[action] == max(all_qs.values())]
        self.optimal_val = float(max(all_qs.values()))  # Update the optimal Q-value
        
        # Return one of the best actions at random in case of a tie
        return random.choice(optimal_actions)

    def update(self, t):
        """
        Update the agent's state and Q-table based on the environment feedback.
        
        Parameters:
        t (int): The current time step.
        """
        # Gather inputs from the environment
        self.next_waypoint = self.planner.next_waypoint()  # Get the next waypoint from the route planner
        inputs = self.env.sense(self)  # Get sensory inputs from the environment
        deadline = self.env.get_deadline(self)  # Get the remaining time to reach the destination
        
        # Increment time step counter
        self.time += 1
        
        # Calculate learning rate (alpha) and discount factor (gamma)
        alpha = 1.0 / self.time
        gamma = 0.35

        # Define the current state based on sensory inputs and the next waypoint
        self.state = (inputs['light'], inputs['oncoming'], inputs['left'], self.next_waypoint)

        # Select the best known action for the current state
        action = self.best_action(self.state)
        
        # Execute the action and get the reward from the environment
        reward = self.env.act(self, action)
        
        # Record errors if the reward is negative
        if reward < 0:
            self.errors += reward

        # Update the Q-value of the (state, action) pair using the Q-learning formula
        self.qs[(self.state, action)] = (1 - alpha) * self.qs.get((self.state, action), 0) \
        + alpha * (reward + gamma * self.optimal_val)
        
        # Debug print statements to observe the agent's behavior
        print("Reward is")
        print(reward)
        print(f"LearningAgent.update(): deadline = {deadline}, inputs = {inputs}, action = {action}, reward = {reward}")

def run():
    """Run the agent for a finite number of trials."""
    total_rew = []
    
    # Set up environment and agent
    e = Environment()  # Create environment (also adds some dummy traffic)
    a = e.create_agent(LearningAgent)  # Create learning agent
    e.set_primary_agent(a, enforce_deadline=True)  # Specify the primary agent to track with enforced deadlines

    # Create and configure the simulator
    sim = Simulator(e, update_delay=0, display=True)  # Create simulator (uses pygame when display=True, if available)
    
    # Run the simulation for a specified number of trials
    for x in range(100):
        sim.run(n_trials=100)  # Run the simulation for 100 trials each time
    # NOTE: To quit midway, press Esc or close pygame window, or hit Ctrl+C on the command-line

if __name__ == '__main__':
    run()  # Execute the run function if the script is run directly
