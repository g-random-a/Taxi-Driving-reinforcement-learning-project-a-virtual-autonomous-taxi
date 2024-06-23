
from environment import GridEnvironment
from agent import QLearningAgent

if __name__ == "__main__":
    env = GridEnvironment(size=5)
    agent = QLearningAgent(env)
    
    agent.train(episodes=1000)
    
    state = env.reset()
    done = False
    while not done:
        env.render()
        action = agent.choose_action(state)
        state, _, done = env.step(action)
    
    print("Final Path:")
    env.render()
