import numpy as np
import matplotlib.pyplot as plt

class HuntingEnv(object):
    """Hunter game environment
        Position (3, )
        ----------
        [in-game_flag, y-position, x-position]
        State (3 * num_hunters + 3 * num_rabbits, )
        ----------
        [[hunter poisition], [rabbits position]]
    """
    def __init__(self, args):
        plt.ion()

        self.shape = args.shape
        self.num_hunters = args.num_hunters
        self.num_rabbits = args.num_rabbits
        self.size = self.num_hunters + self.num_rabbits
        self.dead_agents = 0 # number of hunters and rabbits not in game
        self.step_reward = args.step_reward
        self.catch_reward = args.catch_reward
        self.done_reward = args.done_reward
        self.nS = np.prod(self.shape)
        self.nA = 9
        self.A = np.array([[i, j] for i in [-1, 0, 1] for j in [-1, 0, 1]])
        self.num_actions = self.A.shape[0]
        self.state = None

    def reset(self, location=None):
        """Reset an environment
            Parameters
            ----------
            Returns
            -------
            start state vector: (3 * num_hunters + 3 * num_rabbits)
        """
        if location is None:
            self.state = self._init_env()
            return self.state_to_map(self.state)

        self.state = np.ones((self.size, 3), dtype=np.int)
        self.state[np.arange(self.size), :] = location[np.arange(self.size), :]
        return self.state_to_map(self.state)

    def _init_env(self):
        start = np.random.randint(0, self.shape, size=3 * self.size)
        start[::3] = 1
        start = start.reshape(-1, 3)
        return start


    def state_to_map(self, state):
        state_map = np.zeros((self.size, self.shape, self.shape))
        hunters = state[:self.num_hunters, :]
        rabbits = state[self.num_hunters:self.size, :]
        state_map[np.arange(self.num_hunters), hunters[np.arange(self.num_hunters), 1], hunters[np.arange(self.num_hunters), 2]] = 1
        state_map[np.arange(self.num_rabbits) + self.num_hunters, rabbits[np.arange(self.num_rabbits), 1], rabbits[np.arange(self.num_rabbits), 2]] = -1
        return(state_map)

    def _limit_coordinates(self, coord):
        coord[0] = np.minimum(coord[0], self.shape - 1)
        coord[0] = np.maximum(coord[0], 0)
        coord[1] = np.minimum(coord[1], self.shape - 1)
        coord[1] = np.maximum(coord[1], 0)
        return coord

    def idx_to_act (self, act_idx):
        return self.A[act_idx, :]

    def _array_equal(self, a, b):
        return a[1] == b[1] and a[2] == b[2]

    def _get_reward(self):
        hunters = self.state[:self.num_hunters, :]
        rabbits = self.state[self.num_hunters:self.size, :]
        # Create a map with locations of rabbits marked by non-zero value
        # the xth rabbit with coordinate (a, b) will mark state_map[a, b] with value x + 1
        state_map = np.zeros((self.shape, self.shape), dtype=np.int)
        for i in range(self.num_rabbits):
            state_map[rabbits[i, 1], rabbits[i, 2]] = i + 1

        reward = 0
        catch_flag = 0
        for i in range(self.num_hunters):
            # Get the value on state_map and check if it's marked
            value = state_map[self.state[i, 1], self.state[i, 2]]
            # Marked by some rabbit
            if (value > 0):
                catch_flag += 1
                # Remove the hunter rabbit pair in that location
                self.state[i, 0] = 0
                self.state[self.num_hunters + value - 1, 0] = 0
                # Clear the marked value on state-map
                # So that when two hunters are at the same position
                # Only the first hunter catches the rabbit
                state_map[self.state[i, 1], self.state[i, 2]] = 0
                reward += self.catch_reward
        if catch_flag > 0:
            self.state = self.state[self.state[:, 0] != 0]
            self.num_hunters -= catch_flag
            self.num_rabbits -= catch_flag
            self.size -= 2 * catch_flag
        else:
            reward = self.step_reward
        return reward

    def step(self, act_idx, rabbit_act=None):
        action = self.idx_to_act(act_idx)
        # Clear rabbit actions for non-moving rabbit problem
        if rabbit_act is None:
            action[self.num_hunters:self.size, :] = np.zeros((self.num_rabbits, 2), dtype=np.int)
        # Predict the possible next positions
        predict = self.state[:, 1:] + action
        # Update the state with valid movements
        self.state[:, 1:] = self._limit_coordinates(predict)
        reward = self._get_reward()
        done = False
        if self.num_hunters == 0 or self.num_rabbits == 0:
            done = True
            reward = self.done_reward
        return self.state_to_map(self.state), reward, done
