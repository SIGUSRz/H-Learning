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
        Q_vec = model(state, False)
        action = int(Q_vec.max(1)[1].data.numpy())
        state, reward, done = env.step(action)
        episode_reward += reward
    env.render(animation=(not data_utils.use_cuda))
    time.sleep(0.5)
    env.step(action)
    print("Episode reward: %f" % episode_reward)


def main(args):
    env = CliffWalkingEnv((10, 10))
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
            current_vec = model(np.asarray(
                current_state, dtype=int) * np.ones(1, dtype=int), False)
            action = model.select_action(current_vec)
            steps += 1
            next_state, reward, done = env.step(action)
            total_reward += reward

            memory.push(current_state, action, next_state, reward)
            current_state = next_state
            # When ready to sample
            if len(memory.memory) >= args.batch_size:
                # Sample current state, action, next state, reward from memory
                transitions = np.asarray(memory.sample(args.batch_size))
                terminal = args.grid_shape * args.grid_shape - 1
                current_batch = transitions[:, 0].astype(int)
                action_batch = Variable(torch.from_numpy(
                    transitions[:, 1].astype(int))).type(data_utils.LongTensor).unsqueeze(1)
                next_batch = transitions[:, 2].astype(int)
                reward_batch = Variable(
                    torch.from_numpy(transitions[:, 3])).type(data_utils.FloatTensor)

                # Mark non terminal step
                non_final_mask = data_utils.ByteTensor(
                    (next_batch != terminal).tolist())
                non_final_states = next_batch[next_batch != terminal]
                # Calculate action values
                current_Q_vec = model(
                    current_batch, False).gather(1, action_batch)
                next_Q_vec = Variable(torch.zeros(
                    args.batch_size).type(data_utils.Tensor))
                next_Q_vec[non_final_mask] = model(
                    non_final_states, True).max(1)[0]
                next_Q_vec.volatile = False
                # Calculate target matrix
                target = next_Q_vec * args.gamma + reward_batch
                # Optimize loss function
                loss = F.smooth_l1_loss(current_Q_vec, target)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                # Record
                if (done or steps == 5000):
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
                if (steps == 5000):
                    break
                # break # debug
        print("Reward: ", total_reward)
        print('==============')
        floss.write("Reward: %s\n" % str(total_reward))
        floss.write("==============\n")
        # break #debug
    floss.close()
    render_single_Q(env, model)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_episodes', type=int, default=1000,
                        help='Number of episodes to train')
    parser.add_argument('--gamma', type=float, default=0.95,
                        help='Decay rate of reward function')
    parser.add_argument('--lr', type=float, default=0.1,
                        help='Learning rate of update')
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
