import time
import random
from collections import OrderedDict

from simulator import Simulator

class TrafficLight(object):
    """A traffic light that switches periodically."""

    valid_states = [True, False]  # True = NS open, False = EW open

    def __init__(self, state=None, period=None):
        """
        Initialize a TrafficLight.

        Parameters:
        state (bool): Initial state of the traffic light (True for NS, False for EW).
        period (int): Period of the traffic light switch in time steps.
        """
        self.state = state if state is not None else random.choice(self.valid_states)
        self.period = period if period is not None else random.choice([3, 4, 5])
        self.last_updated = 0  # Time step when the light was last updated

    def reset(self):
        """Reset the traffic light's last updated time."""
        self.last_updated = 0

    def update(self, t):
        """
        Update the traffic light's state based on the current time step.

        Parameters:
        t (int): Current time step.
        """
        if t - self.last_updated >= self.period:
            self.state = not self.state  # Toggle the state
            self.last_updated = t  # Update the last updated time


class Environment(object):
    """Environment within which all agents operate."""

    valid_actions = [None, 'forward', 'left', 'right']
    valid_inputs = {'light': TrafficLight.valid_states, 'oncoming': valid_actions, 'left': valid_actions, 'right': valid_actions}
    valid_headings = [(1, 0), (0, -1), (-1, 0), (0, 1)]  # ENWS
    hard_time_limit = -100  # End trial when deadline reaches this value to avoid deadlocks

    def __init__(self):
        """
        Initialize the environment.
        """
        self.done = False  # Indicates if the trial is done
        self.t = 0  # Time step counter
        self.agent_states = OrderedDict()  # Store agent states
        self.status_text = ""  # Status text for debugging

        # Road network
        self.grid_size = (8, 6)  # (cols, rows)
        self.bounds = (1, 1, self.grid_size[0], self.grid_size[1])
        self.block_size = 100
        self.intersections = OrderedDict()  # Store traffic lights at intersections
        self.roads = []  # Store road connections

        # Initialize intersections with traffic lights
        for x in range(self.bounds[0], self.bounds[2] + 1):
            for y in range(self.bounds[1], self.bounds[3] + 1):
                self.intersections[(x, y)] = TrafficLight()  # A traffic light at each intersection

        # Initialize roads between adjacent intersections
        for a in self.intersections:
            for b in self.intersections:
                if a == b:
                    continue
                if (abs(a[0] - b[0]) + abs(a[1] - b[1])) == 1:  # L1 distance = 1
                    self.roads.append((a, b))

        # Dummy agents
        self.num_dummies = 3  # Number of dummy agents
        for i in range(self.num_dummies):
            self.create_agent(DummyAgent)

        # Primary agent
        self.primary_agent = None  # To be set explicitly
        self.enforce_deadline = False

    def create_agent(self, agent_class, *args, **kwargs):
        """
        Create a new agent and add it to the environment.

        Parameters:
        agent_class (class): The class of the agent to create.
        """
        agent = agent_class(self, *args, **kwargs)
        self.agent_states[agent] = {'location': random.choice(list(self.intersections.keys())), 'heading': (0, 1)}
        return agent

    def set_primary_agent(self, agent, enforce_deadline=False):
        """
        Set the primary agent in the environment.

        Parameters:
        agent (Agent): The primary agent.
        enforce_deadline (bool): Whether to enforce deadlines for the primary agent.
        """
        self.primary_agent = agent
        self.enforce_deadline = enforce_deadline

    def reset(self):
        """Reset the environment for a new trial."""
        self.done = False
        self.t = 0

        # Reset traffic lights
        for traffic_light in self.intersections.values():
            traffic_light.reset()

        # Pick a start and a destination
        start = random.choice(list(self.intersections.keys()))
        destination = random.choice(list(self.intersections.keys()))

        # Ensure starting location and destination are not too close
        while self.compute_dist(start, destination) < 4:
            start = random.choice(list(self.intersections.keys()))
            destination = random.choice(list(self.intersections.keys()))

        start_heading = random.choice(self.valid_headings)
        deadline = self.compute_dist(start, destination) * 5
        print("Environment.reset(): Trial set up with start = {}, destination = {}, deadline = {}".format(start, destination, deadline))

        # Initialize agents
        for agent in self.agent_states.keys():
            self.agent_states[agent] = {
                'location': start if agent is self.primary_agent else random.choice(list(self.intersections.keys())),
                'heading': start_heading if agent is self.primary_agent else random.choice(self.valid_headings),
                'destination': destination if agent is self.primary_agent else None,
                'deadline': deadline if agent is self.primary_agent else None}
            agent.reset(destination=(destination if agent is self.primary_agent else None))

    def step(self):
        """Advance the environment by one time step."""
        # Update traffic lights
        for intersection, traffic_light in self.intersections.items():
            traffic_light.update(self.t)

        # Update agents
        for agent in self.agent_states.keys():
            agent.update(self.t)

        self.t += 1
        if self.primary_agent is not None:
            agent_deadline = self.agent_states[self.primary_agent]['deadline']
            if agent_deadline <= self.hard_time_limit:
                self.done = True
                print("Environment.step(): Primary agent hit hard time limit ({})! Trial aborted.".format(self.hard_time_limit))
            elif self.enforce_deadline and agent_deadline <= 0:
                self.done = True
                print("Environment.step(): Primary agent ran out of time! Trial aborted.")
            self.agent_states[self.primary_agent]['deadline'] = agent_deadline - 1

    def sense(self, agent):
        """
        Sense the state of the environment for the given agent.

        Parameters:
        agent (Agent): The agent sensing the environment.

        Returns:
        dict: A dictionary of sensory inputs.
        """
        assert agent in self.agent_states, "Unknown agent!"

        state = self.agent_states[agent]
        location = state['location']
        heading = state['heading']
        light = 'green' if (self.intersections[location].state and heading[1] != 0) or ((not self.intersections[location].state) and heading[0] != 0) else 'red'

        # Populate oncoming, left, right
        oncoming = None
        left = None
        right = None
        for other_agent, other_state in self.agent_states.items():
            if agent == other_agent or location != other_state['location'] or (heading[0] == other_state['heading'][0] and heading[1] == other_state['heading'][1]):
                continue
            other_heading = other_agent.get_next_waypoint()
            if (heading[0] * other_state['heading'][0] + heading[1] * other_state['heading'][1]) == -1:
                if oncoming != 'left':  # We don't want to override oncoming == 'left'
                    oncoming = other_heading
            elif (heading[1] == other_state['heading'][0] and -heading[0] == other_state['heading'][1]):
                if right != 'forward' and right != 'left':  # We don't want to override right == 'forward or 'left'
                    right = other_heading
            else:
                if left != 'forward':  # We don't want to override left == 'forward'
                    left = other_heading

        return {'light': light, 'oncoming': oncoming, 'left': left, 'right': right}

    def get_deadline(self, agent):
        """
        Get the deadline for the given agent.

        Parameters:
        agent (Agent): The agent to get the deadline for.

        Returns:
        int: The deadline for the agent.
        """
        return self.agent_states[agent]['deadline'] if agent is self.primary_agent else None

    def act(self, agent, action):
        """
        Perform the given action for the agent.

        Parameters:
        agent (Agent): The agent performing the action.
        action (str): The action to perform.

        Returns:
        float: The reward for the action.
        """
        assert agent in self.agent_states, "Unknown agent!"
        assert action in self.valid_actions, "Invalid action!"

        state = self.agent_states[agent]
        location = state['location']
        heading = state['heading']
        light = 'green' if (self.intersections[location].state and heading[1] != 0) or ((not self.intersections[location].state) and heading[0] != 0) else 'red'
        sense = self.sense(agent)

        # Move agent if within bounds and obeys traffic rules
        reward = 0  # Reward/penalty
        move_okay = True
        if action == 'forward':
            if light != 'green':
                move_okay = False
        elif action == 'left':
            if light == 'green' and (sense['oncoming'] == None or sense['oncoming'] == 'left'):
                heading = (heading[1], -heading[0])
            else:
                move_okay = False
        elif action == 'right':
            if light == 'green' or sense['left'] != 'straight':
                heading = (-heading[1], heading[0])
            else:
                move_okay = False

        if move_okay:
            # Valid move (could be null)
            if action is not None:
                # Valid non-null move
                location = ((location[0] + heading[0] - self.bounds[0]) % (self.bounds[2] - self.bounds[0] + 1) + self.bounds[0],
                            (location[1] + heading[1] - self.bounds[1]) % (self.bounds[3] - self.bounds[1] + 1) + self.bounds[1])  # wrap-around
                state['location'] = location
                state['heading'] = heading
                reward = 2.0 if action == agent.get_next_waypoint() else -0.5  # Valid, but is it correct? (as per waypoint)
            else:
                # Valid null move
                reward = 0.0
        else:
            # Invalid move
            reward = -1.0

        if agent is self.primary_agent:
            if state['location'] == state['destination']:
                if state['deadline'] >= 0:
                    reward += 10  # Bonus for reaching destination on time
                self.done = True
                print("Environment.act(): Primary agent has reached destination!")  # [debug]
            self.status_text = "state: {}\naction: {}\nreward: {}".format(agent.get_state(), action, reward)

        return reward

    def compute_dist(self, a, b):
        """
        Compute the L1 distance between two points.

        Parameters:
        a (tuple): The first point (x, y).
        b (tuple): The second point (x, y).

        Returns:
        int: The L1 distance between the points.
        """
        return abs(b[0] - a[0]) + abs(b[1] - a[1])


class Agent(object):
    """Base class for all agents."""

    def __init__(self, env):
        """
        Initialize an agent.

        Parameters:
        env (Environment): The environment the agent interacts with.
        """
        self.env = env
        self.state = None
        self.next_waypoint = None
        self.color = 'cyan'  # Default color

    def reset(self, destination=None):
        """Reset the agent for a new trial."""
        pass

    def update(self, t):
        """Update the agent's state based on the current time step."""
        pass

    def get_state(self):
        """Get the agent's current state."""
        return self.state

    def get_next_waypoint(self):
        """Get the agent's next waypoint."""
        return self.next_waypoint


class DummyAgent(Agent):
    color_choices = ['blue', 'cyan', 'magenta', 'orange']

    def __init__(self, env):
        """
        Initialize a DummyAgent.

        Parameters:
        env (Environment): The environment instance the agent interacts with.
        """
        super(DummyAgent, self).__init__(env)  # Initialize the parent class (Agent)
        self.next_waypoint = random.choice(Environment.valid_actions[1:])  # Randomly choose the next waypoint
        self.color = random.choice(self.color_choices)  # Randomly choose a color for the agent

    def update(self, t):
        """
        Update the dummy agent's state based on the current time step.

        Parameters:
        t (int): The current time step.
        """
        inputs = self.env.sense(self)  # Sense the environment

        action_okay = True  # Assume action is okay initially
        if self.next_waypoint == 'right':
            if inputs['light'] == 'red' and inputs['left'] == 'forward':
                action_okay = False
        elif self.next_waypoint == 'forward':
            if inputs['light'] == 'red':
                action_okay = False
        elif self.next_waypoint == 'left':
            if inputs['light'] == 'red' or (inputs['oncoming'] == 'forward' or inputs['oncoming'] == 'right'):
                action_okay = False

        action = None
        if action_okay:
            action = self.next_waypoint
            self.next_waypoint = random.choice(Environment.valid_actions[1:])  # Choose the next waypoint randomly
        reward = self.env.act(self, action)  # Perform the action and get the reward
        #print "DummyAgent.update(): t = {}, inputs = {}, action = {}, reward = {}".format(t, inputs, action, reward)  # [debug]
        #print "DummyAgent.update(): next_waypoint = {}".format(self.next_waypoint)  # [debug]
