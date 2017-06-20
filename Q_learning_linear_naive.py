import time
import argparse
import numpy as np
from env.cliff_walking import CliffWalkingEnv


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
    print (w)
    while not done:
        env.render()
        time.sleep(0.5)
        state_vec = np.eye(env.nS)[state - 1]
        action = np.argmax(np.dot(state_vec, w))
        state, reward, done = env.step(action)
        episode_reward += reward
    env.render()
    time.sleep(0.5)
    env.step(action)
    print("Episode reward: %f" % episode_reward)


def main(args):
    env = CliffWalkingEnv((5, 4))
    # Linear Mapping from State Space to Action Values of a tuple (S, a)
    w = np.random.uniform(0, 0.1, (env.nS, env.nA))
    epsilon = args.epsilon
    for i in range(args.num_episodes):
        print("episode: %d" % i)
        current_state = env.reset()
        done = False
        while not done:
            # Creating One-hot Vector of Current State
            current_state_vec = np.eye(env.nS)[current_state - 1]
            current_Q = np.dot(current_state_vec, w)
            explore = np.random.choice((0, 1), p=[epsilon, 1 - epsilon])
            if explore:
                action = np.random.choice(env.nA)
            else:
                action = np.argmax(current_Q)
            new_state, reward, done = env.step(action)
            # Creating One-hot Vector of Next State
            new_state_vec = np.eye(env.nS)[new_state - 1]
            next_Q = np.dot(new_state_vec, w)
            # Gradient Update
            delta = args.lr * (reward + args.gamma * np.max(next_Q) -
                               current_Q[action]) * current_state_vec
            w[:, action] += delta
            current_state = new_state
        epsilon *= args.eps_decay
    render_single_Q(env, w)


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
    args = parser.parse_args()
    main(args)
