import time
import argparse
import numpy as np
from env.cliff_walking import CliffWalkingEnv
import data_utils


def render_single_Q(env, w):
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
        env.render(animation=(not data_utils.use_cuda))
        time.sleep(0.5)
        state_vec = np.eye(env.nS)[state - 1]
        action = np.argmax(np.dot(state_vec, w))
        state, reward, done = env.step(action)
        episode_reward += reward
    env.render(animation=(not data_utils.use_cuda))
    time.sleep(0.5)
    env.step(action)
    print("Episode reward: %f" % episode_reward)


def main(args):
    env = CliffWalkingEnv((5, 4))
    # Linear Mapping from State Space to Action Values of a tuple (S, a)
    w = np.random.uniform(0, 0.1, (env.nS, env.nA))
    for i in range(args.num_episodes):
        print("episode: %d" % i)
        current_state = env.reset()
        done = False
        steps = 0
        while not done:
            # Creating One-hot Vector of Current State
            current_state_vec = np.eye(env.nS)[current_state - 1]
            current_Q = np.dot(current_state_vec, w)
            sample = np.random.random()
            eps_threshold = args.eps_end + (args.eps_start - args.eps_end) * \
                np.exp(-1. * steps / args.eps_decay)
            if sample > eps_threshold:
                action = np.argmax(current_Q)
            else:
                action = np.random.choice(env.nA)
            steps += 1
            new_state, reward, done = env.step(action)
            # Creating One-hot Vector of Next State
            new_state_vec = np.eye(env.nS)[new_state - 1]
            next_Q = np.dot(new_state_vec, w)
            # Gradient Update
            delta = args.lr * (reward + args.gamma * np.max(next_Q) -
                               current_Q[action]) * current_state_vec
            w[:, action] += delta
            current_state = new_state
    render_single_Q(env, w)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_episodes', type=int,
                        default=100, help='Number of episodes to train')
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
    args = parser.parse_args()
    main(args)
