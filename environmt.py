
import numpy as np

class GridEnvironment:
    def __init__(self, size=5):
        self.size = size
        self.grid = np.zeros((size, size))
        self.state = (0, 0)
        self.goal = (size - 1, size - 1)
    
    def reset(self):
        self.state = (0, 0)
        return self.state
    
    def step(self, action):
        x, y = self.state
        if action == 0:   # Up
            x = max(0, x - 1)
        elif action == 1: # Down
            x = min(self.size - 1, x + 1)
        elif action == 2: # Left
            y = max(0, y - 1)
        elif action == 3: # Right
            y = min(self.size - 1, y + 1)
        
        self.state = (x, y)
        reward = -1
        done = self.state == self.goal
        if done:
            reward = 0
        return self.state, reward, done

    def render(self):
        grid = np.zeros((self.size, self.size))
        grid[self.state] = 1
        grid[self.goal] = 2
        print(grid)
