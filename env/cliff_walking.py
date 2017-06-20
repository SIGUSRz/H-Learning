import numpy as np
import matplotlib.pyplot as plt


class CliffWalkingEnv(object):
    def __init__(self, shape):
        plt.ion()

        self.shape = shape
        self.nS = np.prod(self.shape)
        self.nA = 9
        self.state = self.reset()
        self.A = [np.array([i, j]) for i in [-1, 0, 1] for j in [-1, 0, 1]]

        self._cliff = np.zeros(self.shape, dtype=np.bool)
        self._cliff[self.shape[0] - 1, 1:-1] = True
        self._goal = (self.shape[0] - 1, self.shape[1] - 1)

    def reset(self):
        self.state = self._init_state()
        return self.state

    def step(self, a, num_agents=1):
        if num_agents > 1:
            a = self._as_to_joint_a(a)
        s = self.state
        position = np.array(np.unravel_index(s, self.shape))
        delta = self._joint_a_to_action(a)
        next_position = position + delta
        next_position = self._limit_coordinates(next_position).astype(int)
        if self._cliff[tuple(next_position)]:
            next_s = self.reset()
            r = -100.0
        else:
            next_s = np.ravel_multi_index(tuple(next_position), self.shape)
            r = -1.0
        self.state = next_s
        done = tuple(next_position) == self._goal
        return next_s, r, done

    def _init_state(self):
        return np.ravel_multi_index((self.shape[0] - 1, 0), self.shape)

    def _limit_coordinates(self, coord):
        coord[0] = min(coord[0], self.shape[0] - 1)
        coord[0] = max(coord[0], 0)
        coord[1] = min(coord[1], self.shape[1] - 1)
        coord[1] = max(coord[1], 0)
        return coord

    def _joint_a_to_action(self, joint_a):
        return self.A[joint_a]

    def _as_to_joint_a(self, as_):
        joint_a = 3 * as_[0] + as_[1]
        return joint_a

    def render(self, animation=False):
        vertical_coor = self.state // self.shape[1]
        horizontal_coor = self.state % self.shape[1]
        print(vertical_coor, horizontal_coor)

        if animation:
            nrows, ncols = self.shape
            image = np.zeros((nrows, ncols))

            image[vertical_coor, horizontal_coor] = 1.
            image[np.where(self._cliff == True)] = 2.
            row_labels = range(nrows)
            col_labels = range(ncols)
            plt.matshow(image)
            plt.xticks(range(ncols), col_labels)
            plt.yticks(range(nrows), row_labels)
            plt.draw()
            plt.pause(0.5)
