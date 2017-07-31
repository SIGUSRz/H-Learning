from env.cliff_walking import CliffWalkingEnv
from env.hunter import HuntingEnv
import argparse
import numpy as np
import model_zoo
import data_utils

def main(args):
    env = HuntingEnv(args)

    locations = np.array([[1, 3, 1], [1, 2, 2], [1, 4, 2], [1, 0, 0]])
    state = env.reset(locations)
    args.num_states = int(env.nS)
    args.num_actions = int(env.nA)
    args.use_cuda = data_utils.use_cuda
    mod_idx = args.num_hunters - 1

    class Model_Set(object):
        def __init__(self, args):
            self.zoo = []
            self.num_hunters = args.num_hunters

            for i in range(self.num_hunters):
                self.zoo.append(model_zoo.MA_hunter(args, i + 1))

    model = Model_Set(args)

    Q_vec= model.zoo[mod_idx](state[np.newaxis, :], False)
    print(Q_vec)
    action = model.zoo[mod_idx].select_action(Q_vec[0])



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_hunters', type=int,
                        default=2, help='Number of hunters in the game')
    parser.add_argument('--num_rabbits', type=int,
                        default=2, help='Number of rabbits in the game')
    parser.add_argument('--step_reward', type=int,
                        default=-1, help='Reward for a normal step')
    parser.add_argument('--catch_reward', type=int,
                        default=10, help='Reward for a normal step')
    parser.add_argument('--done_reward', type=int,
                        default=100, help='Reward for a normal step')
    parser.add_argument('--num_episodes', type=int,
                        default=500, help='Number of episodes to train')
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
                        default=5, help='grid shape')
    parser.add_argument('--batch_size', type=int,
                        default=128, help='Batch Size')
    args = parser.parse_args()

    main(args)
