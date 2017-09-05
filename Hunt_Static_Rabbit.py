import time
import copy
import numpy as np
import argparse
import random
import collections

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from env.hunter import HuntingEnv
import model_zoo
import data_utils


def render_single(model, args):
    """Run policy and paint steps
        Parameters
        ----------
        env: environment object
        model: trained model, model_zoo.Model
        Returns
        -------
        None
    """
    episode_reward = 0
    env = HuntingEnv(args)
    state = env.reset()
    mod_idx = len(model.zoo) - 1
    done = False
    action = None
    steps = 0
    open('hlog.txt', 'w').close()
    hlog = open('hlog.txt', 'a')
    # Wheter to visualize the step

    while (not done and steps < 1000):
        steps += 1
        dead_hunters = []
        dead_rabbits = []
        env.render(animation=(not data_utils.use_cuda))
        time.sleep(0.25)
        current_model = model.zoo[mod_idx]
        current_memo = model.memory[mod_idx]
        current_optim = model.optimizer[mod_idx]
        hlog.write("Hunter: \n")
        hlog.write(np.array_str(state[:args.num_hunters, :]))
        hlog.write("\n")
        if data_utils.use_cuda:
            state_vec = current_model(state[np.newaxis, :], False)
            action = current_model.select_action(state_vec[0])
        else:
            state_vec = current_model(state[np.newaxis, :], False)
            action = current_model.select_action(state_vec[0])
        state, reward, done, dead_hunters, dead_rabbits = env.step(action)
        episode_reward += reward
        if(len(dead_hunters) > 0):
            print('Remove One')
            mod_idx -= len(dead_hunters)
            hlog.write("Dead Rabbit: \n")
            for i in dead_rabbits:
                hlog.write(str(i[0]) + ' ' + np.array_str(i[1]))
                hlog.write("\n")
    env.step(action)
    print("Final reward: %f" % episode_reward)
    f = open('loss.txt', 'a')
    f.write("Final Reward: %s\n" % str(episode_reward))
    f.close()
    hlog.close()


class ReplayMemory(object):
    """Replay memory sample class
        Parameters
        ----------
        capacity: memory capacity, int
        Returns
        -------
        class
    """

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = args
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def get_shape(self, idx):
        return np.array(list(map(lambda x: x.shape[0], np.asarray(self.memory)[:, idx]))).astype(np.int)

    def __len__(self):
        return len(self.memory)


class Model_Set(object):
    def __init__(self, args):
        self.zoo = []
        self.optimizer = []
        self.memory = []
        self.num_hunters = args.num_hunters

        for i in range(self.num_hunters):
            self.zoo.append(model_zoo.MA_hunter(args, i + 1))
            self.memory.append(ReplayMemory(args.memo_capacity))
            self.optimizer.append(optim.RMSprop(self.zoo[i].parameters()))


def main(args):
    env = HuntingEnv(args)
    args.num_states = int(env.nS)
    args.num_actions = int(env.nA)
    args.use_cuda = data_utils.use_cuda

    model = Model_Set(args)
    if data_utils.use_cuda:
        for m in model.zoo:
            m.cuda()
    open('loss.txt', 'w').close()
    floss = open('loss.txt', 'a')

    for i in range(args.num_episodes):
        steps = 0
        total_reward = 0
        print("Episode: %d" % (i + 1))
        floss.write("Episode: %d\n" % (i + 1))
        locations = np.array([[1, 3, 1], [1, 2, 2], [1, 4, 2], [1, 0, 0], [1, 4, 4], [1, 1, 3], [1, 0, 2], [1, 2, 4]])
        current_state = env.reset(locations)
        # current_state = env.reset()
        mod_idx = args.num_hunters - 1
        done = False
        while (not done and steps < 2000):
            # Take and evaluate a new step
            current_model = model.zoo[mod_idx]
            current_memo = model.memory[mod_idx]
            current_optim = model.optimizer[mod_idx]
            # Reshape state vector to [1, num_agents, 2] for 1 sample of state as input
            current_vec = current_model(current_state[np.newaxis, :], False)
            action = current_model.select_action(current_vec[0, :])
            steps += 1
            next_state, reward, done, dead_hunters, dead_rabbits = env.step(
                action)
            # To be implement: push model shifting step into memory
            # ================
            if(len(dead_hunters) > 0):
                print('Remove One')
                mod_idx -= len(dead_hunters)
            else:
                current_memo.push(current_state, action, next_state, reward)
            total_reward += reward

            current_state = next_state
            # When ready to sample
            if len(current_memo.memory) >= args.batch_size:
                # Sample current state, action, next state, reward from memory
                transitions = np.asarray(current_memo.sample(args.batch_size))
                # To be implement: push model shifting step into memory
                # ================
                # current_shape = current_memo.get_shape(0)
                # next_shape = current_memo.get_shape(2)
                # non_final_mask = data_utils.ByteTensor(
                #     (current_shape == next_shape).tolist())
                current_batch = np.vstack(
                    transitions[:, 0]).reshape(-1, current_model.num_hunters, 2)
                action_batch = Variable(torch.from_numpy(
                    np.vstack(transitions[:, 1]))).type(data_utils.LongTensor).unsqueeze(2)
                next_batch = np.vstack(
                    transitions[:, 2]).reshape(-1, current_model.num_hunters, 2)
                reward_batch = Variable(
                    torch.from_numpy(np.vstack(transitions[:, 3]))).type(data_utils.FloatTensor)

                # To be implement: push model shifting step into memory
                # ================
                # Mark non terminal step
                # non_final_states = next_batch[next_batch != terminal]
                # Calculate action values
                current_Q_vec = current_model(
                    current_batch, False).gather(2, action_batch).squeeze(2)
                next_Q_vec = Variable(torch.zeros(
                    args.batch_size, current_model.num_hunters, args.num_actions).type(data_utils.Tensor))
                if not data_utils.use_cuda:
                    next_Q_vec = current_model(
                        next_batch, True).max(2)[0].squeeze(2)
                else:
                    next_Q_vec = current_model(
                        next_batch, True).max(2)[0]
                next_Q_vec.volatile = False
                # Calculate target matrix
                target = next_Q_vec * args.gamma + \
                    reward_batch.expand_as(next_Q_vec)
                # Optimize loss function
                loss = F.smooth_l1_loss(current_Q_vec, target)
                current_optim.zero_grad()
                loss.backward()
                current_optim.step()
                # Record
                if (done or steps >= 2000):
                    if torch.cuda.is_available():
                        print("==============")
                        print("Finished at step: %d" % steps)
                        print("Loss: %s" % str(loss.data.cpu().numpy()))
                        floss.write("==============\n")
                        floss.write("Finished at step: %d\n" % steps)
                        floss.write(("Loss: %s\n" %
                                     str(loss.data.cpu().numpy())))
                    else:
                        print("==============")
                        print("Finished at step: %d" % steps)
                        print("Loss: %s" % str(loss.data.numpy()))
                        floss.write("==============\n")
                        floss.write("Finished at step: %d\n" % steps)
                        floss.write(("Loss: %s\n" % str(loss.data.numpy())))
                # break # debug
        print("Reward: ", total_reward)
        print('==============')
        floss.write("Reward: %s\n" % str(total_reward))
        floss.write("==============\n")
        # break #debug
    floss.close()
    # render_single(model, args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_hunters', type=int,
                        default=4, help='Number of hunters in the game')
    parser.add_argument('--num_rabbits', type=int,
                        default=4, help='Number of rabbits in the game')
    parser.add_argument('--step_reward', type=int,
                        default=-1, help='Reward for a normal step')
    parser.add_argument('--catch_reward', type=int,
                        default=10, help='Reward for catching a rabbit')
    parser.add_argument('--fall_reward', type=int,
                        default=-10, help='Reward for out of map')
    parser.add_argument('--done_reward', type=int,
                        default=100, help='Reward for a normal step')
    parser.add_argument('--num_episodes', type=int,
                        default=100, help='Number of episodes to train')
    parser.add_argument('--hidden_dim', type=int,
                        default=64, help='Hidden layer dimension')
    parser.add_argument('--gamma', type=float, default=0.99,
                        help='Decay rate of reward function')
    parser.add_argument('--lr', type=float, default=0.05,
                        help='Learning rate of update')
    parser.add_argument('--eps_start', type=float,
                        default=0.9, help="Epsilon init value")
    parser.add_argument('--eps_end', type=float,
                        default=0.05, help="Epsilon lower bound")
    parser.add_argument('--eps_decay', type=float, default=200,
                        help='Decay rate of epsilon')
    parser.add_argument('--memo_capacity', type=int,
                        default=10000, help='Memory capacity')
    parser.add_argument('--grid_shape', type=int,
                        default=8, help='grid shape')
    parser.add_argument('--batch_size', type=int,
                        default=32, help='Batch Size')
    args = parser.parse_args()

    main(args)
