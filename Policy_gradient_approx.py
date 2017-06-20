import copy
import time
import argparse
import numpy as np
from env.cliff_walking import CliffWalkingEnv
from model_zoo import *

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim


def render_single_Q(env, model):
    """Run policy and pain steps
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
        env.render()
        time.sleep(0.5)
        state_vec = Variable(model.encode_state(state), volatile=True)
        Q_vec = model.forward(state_vec)
        action = int(Q_vec.max(1)[1].data.numpy())
        state, reward, done = env.step(action)
        episode_reward += reward
    env.render()
    time.sleep(0.5)
    env.step(action)
    print("Episode reward: %f" % episode_reward)


def main(args):
    env = CliffWalkingEnv((5, 4))
    # Linear Mapping from State Space to Action Values of a tuple (S, a)
    args.num_states = int(env.nS)
    args.num_actions = int(env.nA)
    model = LinearApprox(args)
    optimizer = optim.SGD(model.parameters(), lr=args.lr)
    epsilon = args.epsilon
    for i in range(args.num_episodes):
        print("episode: %d" % i)
        current_state = env.reset()
        done = False
        while not done:
            current_state_vec = Variable(model.encode_state(current_state))
            current_Q_vec = model.forward(current_state_vec)
            explore = np.random.choice((0, 1), p=[args.epsilon, 1 - args.epsilon])
            if explore:
                action = np.random.choice(args.num_actions)
            else:
                action = int(current_Q_vec.max(1)[1].data.numpy())
            next_state, reward, done = env.step(action)
            next_state_vec = Variable(model.encode_state(next_state))
            next_Q_vec = np.zeros((1, args.num_actions))
            next_Q = model(next_state_vec)
            next_Q_vec[0, action] = reward + args.gamma * next_Q.max(1)[0].data.numpy()
            next_Q_vec = Variable(torch.Tensor.float(torch.from_numpy(next_Q_vec)))
            # Gradient Update
            criterion = nn.MSELoss()
            loss = criterion(current_Q_vec[0, action], next_Q_vec[0, a])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            current_state = next_state
        epsilon *= args.eps_decay
    render_single_Q(env, model)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_episodes', type=int,
                        required=True, help='Number of episodes to train')
    parser.add_argument('--gamma', type=float, required=True,
                        help='Decay rate of reward function')
    parser.add_argument('--lr', type=float, required=True,
                        help='Learning rate of update')
    parser.add_argument('--epsilon', type=float, required=True,
                        help='Init value for epsilon greedy exploration')
    parser.add_argument('--eps_decay', type=float,
                        required=True, help='Decay rate of epsilon')
    parser.add_argument('--weight_init', type=float, required=True, help='Weight initialization upper bound')
    parser.add_argument('--bias_init', type=float, required=True, help='Bias initialization')
    parser.add_argument('--hidden_dim', type=int, required=True, help='Hidden layer dimension')
    parser.add_argument('--use_cuda', type=bool, required=True, help='Flag for GPU use')
    args = parser.parse_args()
    main(args)
