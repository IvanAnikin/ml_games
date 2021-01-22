

import time
import numpy as np

#from nes_py.wrappers import BinarySpaceToDiscreteSpaceEnv
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT, RIGHT_ONLY
from nes_py.wrappers import JoypadSpace

from OpenAi.SuperMario.Agents.Agent import DQN_Agent
from OpenAi.SuperMario.Agents.wrapper import wrapper




# Parameters
states = (84, 84, 4) # 0-width, 1-height, 2-skip

env = gym_super_mario_bros.make('SuperMarioBros-1-1-v0')
#env = BinarySpaceToDiscreteSpaceEnv(env, RIGHT_ONLY)
env = JoypadSpace(env, SIMPLE_MOVEMENT)
env = wrapper(env, states)

actions = env.action_space.n


