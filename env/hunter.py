import numpy as np
import matplotlib.pyplot as plt

class HuntingEnv():
    def __init__(self, args):
        plt.ion()

        self.shape = args.shape
        self.num_hunters = args.hunters
        self.num_rabbits = args.rabbits
        self.num_agents = args.num_agents
        self.nS = np.prod(self.shape)
        self.nA = 9
        self.state = self.reset()
