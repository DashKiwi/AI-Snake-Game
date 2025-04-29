from agent import VectorizedAgent

def train():
    agent = VectorizedAgent(num_envs=32)
    agent.train()

if __name__ == '__main__':
    train()

