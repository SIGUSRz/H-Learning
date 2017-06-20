import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torch.autograd import Variable
import data_utils


class Model(nn.Module):
    """Base object for a model
        Parameters
        ----------
        args: namespace object
    """

    def __init__(self, args):
        super(Model, self).__init__()
        # self.history_len = args.history_len
        self.num_states = args.num_states
        self.num_actions = args.num_actions
        self.use_cuda = args.use_cuda

    def print_model(self):
        print("<--------Model-------->")

    def _init_weights(self):
        """Weight initialization to be override
            Parameters
            ----------
            Returns
            -------
        """
        pass

    def _reset(self):
        """Must call in every model __init__() function
            Parameters
            ----------
            Returns
            -------
        """
        self._init_weights()
        self.print_model()


class LinearApprox(Model):
    """Basic linear approximation method of Q-learning
        Parameters
        ----------
        args: namespace object
    """

    def __init__(self, args):
        super(LinearApprox, self).__init__(args)
        self.hidden_dim = args.hidden_dim
        self.bias_init = args.bias_init
        self.weight_init = args.weight_init
        self.epsilon = args.epsilon
        self.output = nn.Linear(self.num_states, self.num_actions, bias=False)

        self._reset()

    def _init_weights(self):
        pass

    def forward(self, state):
        state_vec = Variable(self._encode_state(state))
        Q_vec = self.output(state_vec)
        return Q_vec

    def _encode_state(self, state):
        """Create One-hot Vector of Current State
            Parameters
            ----------
            state: input state
            Returns
            -------
            One-hot torch FloatTensor: (1, num_states)
        """
        return torch.eye(self.num_states)[state - 1].view(1, self.num_states)

    def select_action(self, Q_vec, epsilon):
        explore = np.random.choice(
            (0, 1), p=[epsilon, 1 - epsilon])
        if explore:
            return np.random.choice(self.num_actions)
        else:
            return int(Q_vec.max(1)[1].data.numpy())


class MLP(Model):
    def __init__(self, args):
        super(MLP, self).__init__(args)
        self.epsilon = args.epsilon
        self.fc1 = nn.Linear(self.num_states, self.hidden_dim)
        self.fc2 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.output = nn.Linear(self.hidden_dim, self.num_actions)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

        self._reset()

    def _init_weights(self):
        pass

    def forward(self, state):
        state_vec = Variable(self._encode_state(state))
        state_vec = self.relu(self.fc1(state_vec))
        state_vec = self.relu(self.fc2(state_vec))
        Q_vec = self.output(state_vec)
        return Q_vec, self._eps_greedy_select(Q_vec)

    def _encode_state(self, state):
        """Create One-hot Vector of Current State
            Parameters
            ----------
            state: input state
            Returns
            -------
            One-hot torch FloatTensor: (1, num_states)
        """
        return torch.eye(self.num_states)[state - 1].view(1, self.num_states)

    def _eps_greedy_select(self, Q_vec):
        explore = np.random.choice(
            (0, 1), p=[self.epsilon, 1 - self.epsilon])
        if explore:
            return np.random.choice(self.num_actions)
        else:
            return int(Q_vec.max(1)[1].data.numpy())


class DQN(Model):
    def __init__(self, args):
        super(DQN, self).__init__(args)
        self.grid_shape = (args.grid_shape, args.grid_shape)
        self.conv1 = nn.Conv2d(1, 4, kernel_size=3, stride=1)
        self.bn1 = nn.BatchNorm2d(4)
        self.conv2 = nn.Conv2d(4, 8, kernel_size=3, stride=1)
        self.bn2 = nn.BatchNorm2d(8)
        self.conv3 = nn.Conv2d(8, 8, kernel_size=3, stride=1)
        self.bn3 = nn.BatchNorm2d(8)
        self.output = nn.Linear((self.grid_shape[0] - 6) ** 2 * 8, 9)
        self.eps_start = args.eps_start
        self.eps_end = args.eps_end
        self.eps_decay = args.eps_decay
        self._steps = 0

        self._reset()

    def _init_weights(self):
        pass

    def forward(self, state):
        x = Variable(torch.from_numpy(self._encode_state(state))).type(data_utils.Tensor)
        if self.use_cuda:
            x.cuda()
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        Q_vec = self.output(x.view(x.size(0), -1))
        return Q_vec

    def _encode_state(self, state):
        encoded_state = np.zeros(
            (state.shape[0], 1, self.grid_shape[0], self.grid_shape[1]), dtype=int)
        state_index = np.unravel_index(state, self.grid_shape)
        # encoded_state[:, 0, -1, :] = -1.
        # encoded_state[:, 0, -1, 0] = 0.
        # encoded_state[:, 0, -1, -1] = 0.
        if np.asarray(state_index[0]) != ():
            encoded_state[np.arange(len(state_index[0])),
                          0, state_index[0], state_index[1]] = 1.
        else:
            encoded_state[np.arange(1), 0, state_index[0], state_index[1]] = 1.
        return encoded_state

    def select_action(self, Q_vec):
        sample = np.random.random()
        eps_threshold = self.eps_end + (self.eps_start - self.eps_end) * \
            np.exp(-1. * self._steps / self.eps_decay)
        self._steps += 1
        if sample > eps_threshold:
            if self.use_cuda:
                return int(Q_vec.data.cpu().max(1)[1].numpy())
            else:
                return int(Q_vec.data.max(1)[1].numpy())
        else:
            return int(np.random.choice(self.num_actions))


class Policy(Model):
    def __init__(self, args):
        super(Policy, self).__init__(args)
        self.to_score = nn.Sequential(nn.Linear(self.num_states, self.hidden_dim), nn.Tanh(
        ), nn.Linear(self.hidden_dim, self.num_actions))
        self.to_value = nn.Sequential(nn.Linear(self.num_states, self.hidden_dim), nn.Tanh(
        ), nn.Linear(self.hidden_dim, self.num_actions))

        self._reset()

    def _init_weights(self):
        pass

    def forward(self, state):
        x = self._encode_state(state)
        score = self.to_score(x)
        value = self.to_value(x)
        prob_score = F.softmax(score)
        action = self._select_action(prob_score).detach()
        log_prob = F.log_softmax(score)
        log_prob = torch.gather(log_prob, 1, action)
        return log_prob, action.data[0, 0], value

    def _select_action(self, prob_score):
        """Sample action from multinomial distribution
            Parameters
            ----------
            prob_score: probability score of actions, FloatTensor
            Returns
            -------
            action matrix
        """
        return torch.multinomial(prob_score, 1)

    def _encode_state(self, state):
        """Create One-hot Vector of Current State
            Parameters
            ----------
            state: input state
            Returns
            -------
            One-hot torch FloatTensor: (1, num_states)
        """
        return torch.eye(self.num_states)[state - 1].view(1, self.num_states)
