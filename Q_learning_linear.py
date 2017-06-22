import time
import argparse
import numpy as np
from env.cliff_walking import CliffWalkingEnv
import model_zoo
import data_utils

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim


def render_single_Q(env, model):
    """Run policy and paint steps
        Parameters
        ----------
        env: environment object
        w: learned parameter of Q, tensor (env.nS, env.nA)
        Returns
        -------
    """
    episode_reward = 0
    state = env.reset()
    done = False
    while not done:
        env.render(animation=(not data_utils.use_cuda))
        time.sleep(0.5)
        Q_vec = model(state)
        action = int(Q_vec.max(1)[1].data.numpy())
        state, reward, done = env.step(action)
        episode_reward += reward
    env.render(animation=(not data_utils.use_cuda))
    time.sleep(0.5)
    env.step(action)
    print("Episode reward: %f" % episode_reward)


def main(args):
    env = CliffWalkingEnv((args.grid_shape, args.grid_shape))
    # Linear Mapping from State Space to Action Values of a tuple (S, a)
    args.num_states = int(env.nS)
    args.num_actions = int(env.nA)
    model = model_zoo.LinearApprox(args)
    optimizer = optim.SGD(model.parameters(), lr=args.lr)
    for i in range(args.num_episodes):
        print("episode: %d" % i)
        current_state = env.reset()
        done = False
        while not done:
            current_Q_vec = model(current_state)
            action = model.select_action(current_Q_vec)
            next_state, reward, done = env.step(action)
            next_Q = model(next_state)
            next_Q_vec = np.zeros((1, args.num_actions))
            next_Q_vec[0, action] = reward + \
                args.gamma * next_Q.max(1)[0].data.numpy()
            next_Q_vec = Variable(torch.Tensor.float(
                torch.from_numpy(next_Q_vec)))
            # Gradient Update
            criterion = nn.MSELoss()
            loss = criterion(current_Q_vec[0, action], next_Q_vec[0, action])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            current_state = next_state
    render_single_Q(env, model)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_episodes', type=int, default=1000,
                        help='Number of episodes to train')
    parser.add_argument('--gamma', type=float, default=0.95,
                        help='Decay rate of reward function')
    parser.add_argument('--lr', type=float, default=0.1,
                        help='Learning rate of update')
    parser.add_argument('--grid_shape', type=int,
                        default=10, help='grid shape')
    parser.add_argument('--eps_start', type=float,
                        default=0.9, help="Epsilon init value")
    parser.add_argument('--eps_end', type=float,
                        default=0.05, help="Epsilon lower bound")
    parser.add_argument('--eps_decay', type=float, default=200,
                        help='Decay rate of epsilon')
    parser.add_argument('--hidden_dim', type=int, default=32,
                        help='Hidden layer dimension')
    args = parser.parse_args()
    main(args)
