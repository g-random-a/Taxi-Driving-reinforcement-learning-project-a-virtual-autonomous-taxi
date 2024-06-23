import random

class RoutePlanner(object):
    """Silly route planner that is meant for a perpendicular grid network."""

    def __init__(self, env, agent):
        """
        Initialize the RoutePlanner.

        Parameters:
        env (Environment): The environment instance.
        agent (Agent): The agent for which the route is being planned.
        """
        self.env = env
        self.agent = agent
        self.destination = None

    def route_to(self, destination=None):
        """
        Set the destination for the agent.

        Parameters:
        destination (tuple): The destination coordinates (x, y). If None, choose a random destination.
        """
        # Choose a random destination if none is provided
        self.destination = destination if destination is not None else random.choice(list(self.env.intersections.keys()))
        print("RoutePlanner.route_to(): destination = {}".format(self.destination))  # [debug]

    def next_waypoint(self):
        """
        Determine the next waypoint for the agent to reach the destination.

        Returns:
        str: The direction the agent should take next ('forward', 'left', 'right', or None).
        """
        # Get the agent's current location and heading
        location = self.env.agent_states[self.agent]['location']
        heading = self.env.agent_states[self.agent]['heading']
        
        # Calculate the delta between the current location and the destination
        delta = (self.destination[0] - location[0], self.destination[1] - location[1])
        
        # Determine the next direction to take based on the delta
        if delta[0] == 0 and delta[1] == 0:
            return None  # Destination reached
        elif delta[0] != 0:  # East-West difference
            if delta[0] * heading[0] > 0:  # Facing correct East-West direction
                return 'forward'
            elif delta[0] * heading[0] < 0:  # Facing opposite East-West direction
                return 'right'  # Long U-turn
            elif delta[0] * heading[1] > 0:
                return 'left'
            else:
                return 'right'
        elif delta[1] != 0:  # North-South difference (turn logic is slightly different)
            if delta[1] * heading[1] > 0:  # Facing correct North-South direction
                return 'forward'
            elif delta[1] * heading[1] < 0:  # Facing opposite North-South direction
                return 'right'  # Long U-turn
            elif delta[1] * heading[0] > 0:
                return 'right'
            else:
                return 'left'
