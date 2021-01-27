

import gym_super_mario_bros
import gym
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from nes_py.wrappers import JoypadSpace


from matplotlib import animation
import matplotlib.pyplot as plt
from PIL import Image

import Clean_Results.Agents.visualisation_agent as visualiser
from OpenAi.SuperMario.Agents.wrapper import wrapper


env_name = 'SuperMarioBros-1-1-v0'
training_version_name = 'Deep Q-Learning' # Double
movement_type = 'SIMPLE_MOVEMENT'
states = (84, 84, 4)        # 0-width, 1-height, 2-skip (learn each + 1)

env = gym_super_mario_bros.make(env_name)
env = JoypadSpace(env, SIMPLE_MOVEMENT)                     # env = BinarySpaceToDiscreteSpaceEnv(env, RIGHT_ONLY)
env = wrapper(env, states)



done = True
state = env.reset()

frames = []

for step in range(500):

    action = env.action_space.sample()
    state, reward, done, info = env.step(action)

    frames.append(Image.fromarray(env.render(mode='rgb_array')))
    env.render()

    if done:
        state = env.reset()

    if step % 100 == 0: print("step: {}".format(step))

with open('openai_gym.gif', 'wb') as f:  # change the path if necessary
    im = Image.new('RGB', frames[0].size)
    im.save(f, save_all=True, append_images=frames)

env.close()
