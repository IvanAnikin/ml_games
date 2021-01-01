

import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from nes_py.wrappers import JoypadSpace

import Clean_Results.Agents.visualisation_agent as visualiser




env = gym_super_mario_bros.make('SuperMarioBros-v0')
env = JoypadSpace(env, SIMPLE_MOVEMENT)



done = True
state = env.reset()

for step in range(5000):

    action = env.action_space.sample()
    state, reward, done, info = env.step(action)

    env.render()
    if done:
        state = env.reset()

env.close()