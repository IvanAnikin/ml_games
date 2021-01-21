
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT, RIGHT_ONLY
from nes_py.wrappers import JoypadSpace

import numpy as np
import time

import tensorflow as tf

import OpenAi.SuperMario.Agents.Agent as Agents_file
import OpenAi.SuperMario.Agents.hyperparameters as hp


episodes = 10
SHOW_EVERY = 20

# Q Learning Agent Parameters
ALPHA = 0.1          # Learning rate
MIN_EPSILON = 0.05   # Random move probability
GAMMA = 0.95         # Discount factor

num_hidden = 128
show_model = False

seed = 42

env_name = "SuperMarioBros-v0"
training_version_name = "Actor_Critic_1"
movement_type = "SIMPLE_MOVEMENT"

env = gym_super_mario_bros.make(env_name)
env = JoypadSpace(env, SIMPLE_MOVEMENT)
env.seed(seed)                                          # SEED -- ?

num_actions = env.action_space.n


Agent = Agents_file.Q_Learning(env=env, gamma=GAMMA, alpha=ALPHA, show_model=show_model, num_hidden=num_hidden)

step = 0

for episode in range(episodes):
    state = env.reset()
    info = 0

    episode_reward = 0
    start = time.time()

    while (True):
        step += 1

        state = tf.convert_to_tensor(state)
        state = tf.expand_dims(state, 0)

        action = Agent.act(state, info)
        #action = np.argmax(action_probs)

        new_state, reward, done, info = env.step(action)
        episode_reward += reward

        if(step % SHOW_EVERY == 0 and step != 0):
            template = "running reward: {:.2f} at episode {} || x_pos: {} || steps: {} || time from episode start: {}"
            print(template.format(episode_reward/step, episode, info['x_pos'], step, time.time() - start))

        if info['flag_get']:
            print('!!! WE GOT THE FLAG !!!')
            flag = True

        if info['x_pos'] >= hp.LEVEL_WIN_DIST:
            print('!!! LEVEL {} Solved !!!'.format(1))

        if done:
            break

        # Learn
        Agent.learn(state=state, reward=reward, next_state=new_state, done=done)


        state = new_state



    # VISUALISATION

    #template = "running reward: {:.2f} at episode {} || x_pos: {} || steps: {} || episode time: {}"
    #rint(template.format(running_reward, episode, info['x_pos'], step, time.time() - start))
    #  print("Step {:>6}, action {:>2} (#{:>2}), gave reward {:>6}, score {:>6} and max score {:>6}, life {:>2} and level {:>2}.".format(step, env0.actions[action], action, reward, info['score'], max_seen_score, info['life'], info['level']))  # DEBUG