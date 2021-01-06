
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT, RIGHT_ONLY
from nes_py.wrappers import JoypadSpace

import tensorflow as tf

import os
import numpy as np
import matplotlib.pyplot as plt

import OpenAi.SuperMario.Agents.Agent as Agent_file
import Clean_Results.Agents.storage_agent as storage_agent
import OpenAi.SuperMario.Agents.visualisation_agent as visualiser

# from example:
# https://github.com/yingshaoxo/ML/tree/master/12.reinforcement_learning_with_mario_bros


episodes = 30000
stats_every = 100
#show_every = 200
#show_length = 50
#show_until = 50
show_model = True
load_model = True

epsilon = 0.1
#end_epsilon_decaying_position = 2
#START_EPSILON_DECAYING = 1
#END_EPSILON_DECAYING = episodes // end_epsilon_decaying_position
#epsilon_decay_value = epsilon / (END_EPSILON_DECAYING - START_EPSILON_DECAYING)



env_name = "SuperMarioBros-v0"
training_version_name = "Simple_NN"
env = gym_super_mario_bros.make(env_name)
#env = JoypadSpace(env, SIMPLE_MOVEMENT)
env = JoypadSpace(env, RIGHT_ONLY)
# env.action_space.sample() = numbers, for example, 0,1,2,3...
# state = RGB of raw picture; is a numpy array with shape (240, 256, 3)
# reward = int; for example, 0, 1 ,2, ...
# done = False or True
# info = {'coins': 0, 'flag_get': False, 'life': 3, 'score': 0, 'stage': 1, 'status': 'small', 'time': 400, 'world': 1, 'x_pos': 40}


EPISODES_NAME = "env-{}__v-{}__ep-{}__stats-{}__episodes".format(env_name, training_version_name, episodes, stats_every)
REWARDS_NAME = "env-{}__v-{}__ep-{}__stats-{}__rewards".format(env_name, training_version_name, episodes, stats_every)
episodes = 10000
model_file_path = "./env-{}__v-{}__ep-{}__stats-{}__nn_model.HDF5".format(env_name, training_version_name, episodes, stats_every)

visualiser.show_env_props(env)

print("episodes: ", episodes)
print("stats_every: ", stats_every)
print("epsilon: ", epsilon)
print("REWARDS_NAME: ", REWARDS_NAME)
print("load_model: ", load_model,  " || show_model: ", show_model)


Agent = Agent_file.Agent_Simple_NN(env=env, model_file_path=model_file_path, load_model=load_model, show_model=show_model)


ep_reward_list = []
avg_reward_list = []
ep_stats = []

done = True
last_state = None
identity = np.identity(env.action_space.n) # for quickly get a hot vector, like 0001000000000000


#for ep in range(episodes):

prev_state = env.reset()
episodic_reward = 0
avg_ep_reward = 0
avg_reward = 0
ep_trained = 0

episodes = 30000
for step in range(episodes):
    if done:
        state = env.reset()
        print("done=True on step: ", step)

    action = Agent.get_action(prev_state=prev_state, epsilon=epsilon)
    state, reward, done, info = env.step(action)

    # save more rewards for batch training - if reward of batch better than other average rewards -> train on it
    if reward > avg_reward: #reward > 0:
        Agent.models.model.train_on_batch(x=np.expand_dims(prev_state, axis=0), y=identity[action: action + 1])
        ep_trained += 1

    prev_state = state

    episodic_reward += reward
    if step % stats_every == 0 and step != 0:
        avg_ep_reward = episodic_reward/stats_every
        avg_reward = (avg_reward + avg_ep_reward) / 2
        ep_reward_list.append(avg_ep_reward)
        ep_stats.append(step)
        print("step: ", step, " || avg_ep_reward: ", avg_ep_reward, " || avg_reward: ", avg_reward, " || ep_trained: ", ep_trained)
        episodic_reward = 0
        ep_trained = 0
    #env.render()
    # if step <= show_until: env.render()
    # if step % show_every == 0 and step != 0:
    #    print("step: ", step)

model_file_path_2 = "./env-{}__v-{}__ep-{}__stats-{}__nn_model_2.HDF5".format(env_name, training_version_name, episodes, stats_every)
Agent.models.model.save(model_file_path_2)
#Agent.models.model.save(model_file_path)

# save ep_reward_list array
storage_agent.save_np(name=EPISODES_NAME, data=np.array(ep_stats))
storage_agent.save_np(name=REWARDS_NAME, data=np.array(ep_reward_list))











#ep_reward_list.append(episodic_reward)

# Mean of last 40 episodes
#avg_reward = np.mean(ep_reward_list[-stats_every:])
#print("Episode * {} * Avg Reward is ==> {}".format(ep, avg_reward))
#avg_reward_list.append(avg_reward)

#if END_EPSILON_DECAYING >= ep >= START_EPSILON_DECAYING:
    #epsilon -= epsilon_decay_value
    #print("")


#plt.plot(avg_reward_list)
#plt.xlabel("Episode")
#plt.ylabel("Avg. Epsiodic Reward")
#lt.show()





#storage_agent.save_np(name=NAME, data=np.array(avg_reward_list))


