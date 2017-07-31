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

        self.shape = args.grid_shape
        self.num_agents = args.num_hunters
        self.step_reward = args.step_reward
        self.catch_reward = args.catch_reward
        self.fall_reward = args.fall_reward
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
        open('log.txt', 'w').close()
        self.file = open('log.txt', 'a')
        self.num_hunters = self.num_agents
        self.num_rabbits = self.num_agents
        self.size = self.num_hunters + self.num_rabbits
        if location is None:
            self.state = self._init_env()
            # return self.state_to_map(self.state)
            return self.state[:self.num_hunters, 1:]

        self.state = np.ones((self.size, 3), dtype=np.int)
        self.state[np.arange(self.size), :] = location[np.arange(self.size), :]
        # return self.state_to_map(self.state)
        return self.state[:self.num_hunters, 1:]

    def _init_env(self):
        start = np.random.randint(0, self.shape, size=3 * self.size)
        start[::3] = 1
        start = start.reshape(-1, 3)
        return start

    def state_to_map(self, state):
        # state_map = np.zeros((self.size, self.shape, self.shape))
        # hunters = state[:self.num_hunters, :]
        # rabbits = state[self.num_hunters:self.size, :]
        # state_map[np.arange(self.num_hunters), hunters[np.arange(self.num_hunters), 1], hunters[np.arange(self.num_hunters), 2]] = 1.
        # state_map[np.arange(self.num_rabbits) + self.num_hunters, rabbits[np.arange(self.num_rabbits), 1], rabbits[np.arange(self.num_rabbits), 2]] = 2.
        # return(state_map)
        state_map = np.zeros((self.shape, self.shape))
        hunters = state[:self.num_hunters, :]
        rabbits = state[self.num_hunters:self.size, :]
        state_map[hunters[np.arange(self.num_hunters), 1],
                  hunters[np.arange(self.num_hunters), 2]] = 1.
        state_map[rabbits[np.arange(self.num_rabbits), 1],
                  rabbits[np.arange(self.num_rabbits), 2]] = 2.
        return(state_map)

    def _limit_coordinates(self, coord):
        up_flag = coord < self.shape
        down_flag = coord >= 0
        return np.all(np.logical_and(up_flag, down_flag), axis=1)

    def idx_to_act(self, act_idx):
        return self.A[act_idx, :]

    def _array_equal(self, a, b):
        return a[1] == b[1] and a[2] == b[2]

    def _get_reward(self, mask):
        hunters = self.state[:self.num_hunters, :]
        rabbits = self.state[self.num_hunters:self.size, :]
        dead_hunters = []
        dead_rabbits = []
        # Create a map with locations of rabbits marked by non-zero value
        # the xth rabbit with coordinate (a, b) will mark state_map[a, b] with value x + 1
        state_map = np.zeros((self.shape, self.shape), dtype=np.int)
        for i in range(self.num_rabbits):
            state_map[rabbits[i, 1], rabbits[i, 2]] = i + 1

        reward = 0
        catch_flag = 0
        num_fall = mask[mask == False].shape[0]
        for i in range(self.num_hunters):
            # Get the value on state_map and check if it's marked
            value = state_map[self.state[i, 1], self.state[i, 2]]
            # Marked by some rabbit
            if (value > 0):
                catch_flag += 1
                # Remove the hunter rabbit pair in that location
                self.state[i, 0] = 0
                self.state[self.num_hunters + value - 1, 0] = 0
                dead_hunters.append((i, self.state[i, 1:]))
                dead_rabbits.append(
                    (self.num_hunters + value - 1, self.state[self.num_hunters + value - 1, 1:]))
                # Clear the marked value on state-map
                # So that when two hunters are at the same position
                # Only the first hunter catches the rabbit
                state_map[self.state[i, 1], self.state[i, 2]] = 0
        if catch_flag > 0:
            self.state = self.state[self.state[:, 0] != 0]
            self.num_hunters -= catch_flag
            self.num_rabbits -= catch_flag
            self.size -= 2 * catch_flag
            reward += self.catch_reward * catch_flag
        reward += self.step_reward * \
            (self.num_hunters - catch_flag - num_fall) + \
            self.fall_reward * num_fall
        return reward, dead_hunters, dead_rabbits

    def step(self, act_idx, rabbit_act=None):
        dead_hunters = None
        dead_rabbits = None
        action = self.idx_to_act(act_idx)
        self.file.write("Hunter: \n")
        self.file.write(np.array_str(self.state[:self.num_hunters, :]))
        self.file.write("\n")
        self.file.write("Action: \n")
        self.file.write(np.array_str(action))
        self.file.write("\n")
        self.file.write("Rabbit: \n")
        self.file.write(np.array_str(self.state[self.num_hunters:, :]))
        self.file.write("\n")
        # Clear rabbit actions for non-moving rabbit problem
        if rabbit_act is None:
            # Predict the possible next positions
            predict = self.state[:self.num_hunters, 1:] + action
            # Update the state with valid movements
            mask = self._limit_coordinates(predict)
            action[np.logical_not(mask)] = [0, 0]
            self.state[:self.num_hunters, 1:] += action
            reward, dead_hunters, dead_rabbits = self._get_reward(mask)
            done = False
            if self.num_hunters == 0 or self.num_rabbits == 0:
                done = True
                reward = self.done_reward
                self.file.close()
            # return self.state_to_map(self.state), reward, done, dead_hunters, dead_rabbits
            return self.state[:self.num_hunters, 1:], reward, done, dead_hunters, dead_rabbits

    def render(self, animation=False):
        if animation:
            image = self.state_to_map(self.state)
            row_labels = range(self.shape)
            col_labels = range(self.shape)
            plt.matshow(image)
            plt.xticks(range(self.shape), col_labels)
            plt.yticks(range(self.shape), row_labels)
            plt.draw()
            plt.pause(0.5)
