import copy

def deepcopy_agent(agent):
    # Create a deep copy of the agent, including model and trainer
    new_agent = copy.deepcopy(agent)
    return new_agent
