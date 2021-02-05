

import gym
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT, RIGHT_ONLY
from nes_py.wrappers import JoypadSpace

from OpenAi.Altari_Games_GAI.Agents.wrapper import wrapper






env_name = 'SpaceInvaders-v0'
training_version_name = 'Deep Q-Learning' # Double


states = (84, 84, 4)


env = gym.make(env_name)
#env = JoypadSpace(env, SIMPLE_MOVEMENT)                     # env = BinarySpaceToDiscreteSpaceEnv(env, RIGHT_ONLY)
env = wrapper(env, states)


env.reset()

while True:

    action = env.action_space.sample()

    state, reward, done, info = env.step(action)

    if(done):
        break