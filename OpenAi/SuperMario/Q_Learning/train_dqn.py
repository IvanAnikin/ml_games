

import time
import numpy as np

#from nes_py.wrappers import BinarySpaceToDiscreteSpaceEnv
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT, RIGHT_ONLY
from nes_py.wrappers import JoypadSpace

from OpenAi.SuperMario.Agents.Agent import DQN_Agent
from OpenAi.SuperMario.Agents.wrapper import wrapper




# Parameters
states = (84, 84, 4)    # 0-width, 1-height, 2-skip
num_hidden = 128
episodes = 1000         # 10000



env = gym_super_mario_bros.make('SuperMarioBros-1-1-v0')
env = JoypadSpace(env, SIMPLE_MOVEMENT)                     # env = BinarySpaceToDiscreteSpaceEnv(env, RIGHT_ONLY)
env = wrapper(env, states)

actions = env.action_space.n


agent = DQN_Agent(env=env, states=states, num_hidden=num_hidden)


# Variables
rewards = []
step = 0
# Timing
start = time.time()


# Main loop
for e in range(episodes):

    state = env.reset()

    total_reward = 0
    iter = 0

    while True:
        # env.render()

        # Run agent
        action = agent.run(state=state)

        # Perform action
        next_state, reward, done, info = env.step(action=action)

        # Remember transition
        agent.add(experience=(state, next_state, action, reward, done))

        # Update agent
        agent.learn()

        # Total reward
        total_reward += reward

        # Update state
        state = next_state

        # Increment
        iter += 1

        # If done break loop
        if done or info['flag_get']:
            break


