
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT, RIGHT_ONLY
from nes_py.wrappers import JoypadSpace

import numpy as np

import tensorflow as tf

import OpenAi.SuperMario.Agents.Agent as Agents_file
import OpenAi.SuperMario.Agents.hyperparameters as hp


episodes = 10
# Q Learning Agent Parameters
ALPHA = 0.1          # Learning rate
MIN_EPSILON = 0.05   # Random move probability
GAMMA = 0.95         # Discount factor

num_hidden = 128
show_model = True

seed = 42

env_name = "SuperMarioBros-v0"
training_version_name = "Actor_Critic_1"
movement_type = "SIMPLE_MOVEMENT"

env = gym_super_mario_bros.make(env_name)
env = JoypadSpace(env, SIMPLE_MOVEMENT)
env.seed(seed)                                          # SEED -- ?

num_actions = env.action_space.n


Agent = Agents_file.Q_Learning(env=env, gamma=GAMMA, alpha=ALPHA, show_model=show_model, num_hidden=num_hidden)



for episode in range(episodes):
    state = env.reset()

    while (True):

        state = tf.convert_to_tensor(state)
        state = tf.expand_dims(state, 0)

        action_probs = Agent.act(state)
        # Sample action from action probability distribution
        #action = np.random.choice(num_actions, p=np.squeeze(action_probs))
        action = np.argmax(action_probs)

        new_state, reward, done, info = env.step(action)

        if info['flag_get']:
            print('!!! WE GOT THE FLAG !!!')
            flag = True

        if info['x_pos'] >= hp.LEVEL_WIN_DIST:
            print('!!! LEVEL {} Solved !!!'.format(1))

        if done:
            break


        # Learn
        Agent.learn(state=state, action_probs=action_probs, reward=reward, next_state=new_state, done=done)






        state = new_state