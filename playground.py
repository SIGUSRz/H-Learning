from env.cliff_walking import CliffWalkingEnv
from env.hunter import HuntingEnv
import numpy as np
env = CliffWalkingEnv((5, 5))
class Obj(object):
    pass
args = Obj()
args.shape = 5
args.num_hunters = 2
args.num_rabbits = 2
args.step_reward = -1
args.catch_reward = 10
henv = HuntingEnv(args)

locations = np.array([[1, 3, 1], [1, 2, 2], [1, 4, 2], [1, 0, 0]])
state = henv.reset(locations)
print("current state")
print(state)
# state = henv.reset()
action = [2, 3, 4, 5]
print("action")
print(action)

state, reward, done = henv.step(action)
print("next state")
print(state)
print("reward")
print(reward)
