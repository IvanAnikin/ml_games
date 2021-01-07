
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT, RIGHT_ONLY
from nes_py.wrappers import JoypadSpace

import tensorflow as tf

import numpy as np
import matplotlib.pyplot as plt
import tqdm

import OpenAi.SuperMario.Agents.Agent as Agent_file
import OpenAi.SuperMario.Agents.Models as Models_file
import Clean_Results.Agents.storage_agent as storage_agent
import OpenAi.SuperMario.Agents.visualisation_agent as visualiser




episodes = 5000
stats_every = 100
show_model = True
load_model = True

epsilon = 0.1
num_hidden_units = 128


env_name = "SuperMarioBros-v0"
training_version_name = "Actor_Critic_1"
env = gym_super_mario_bros.make(env_name)
env = JoypadSpace(env, RIGHT_ONLY)


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



num_actions = env.action_space.n  # 2
num_hidden_units = 128
gamma = 0.99


agent = Agent_file.Actor_Critic_1(env=env, gamma=gamma, num_hidden_units=num_hidden_units)





ep_reward_list = []
avg_reward_list = []
ep_stats = []

done = True
last_state = None
identity = np.identity(env.action_space.n) # for quickly get a hot vector, like 0001000000000000


prev_state = env.reset()
episodic_reward = 0
avg_ep_reward = 0
avg_reward = 0
ep_trained = 0




max_episodes = 10000
max_steps_per_episode = 1000

# Cartpole-v0 is considered solved if average reward is >= 195 over 100
# consecutive trials
reward_threshold = 195
running_reward = 0

# Discount factor for future rewards
gamma = 0.99



with tqdm.trange(max_episodes) as t:
  for i in t:
    initial_state = tf.constant(env.reset(), dtype=tf.float32)
    episode_reward = int(agent.train_step(
        initial_state, max_steps_per_episode))

    running_reward = episode_reward*0.01 + running_reward*.99

    t.set_description(f'Episode {i}')
    t.set_postfix(
        episode_reward=episode_reward, running_reward=running_reward)

    # Show average episode reward every 10 episodes
    if i % 10 == 0:
      pass # print(f'Episode {i}: average reward: {avg_reward}')

    if running_reward > reward_threshold:
        break

print(f'\nSolved at episode {i}: average reward: {running_reward:.2f}!')



'''

for step in range(episodes):
    if done:
        state = env.reset()
        print("done=True on step: ", step)

    action, critic = model.call(inputs=prev_state, epsilon=epsilon ) #INPUTS -- ?
    state, reward, done, info = env.step(action)


    if reward > avg_reward:
        model.train_on_batch(x=np.expand_dims(prev_state, axis=0), y=identity[action: action + 1])
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
'''






#model.save(model_file_path) #model save - ?

storage_agent.save_np(name=EPISODES_NAME, data=np.array(ep_stats))
storage_agent.save_np(name=REWARDS_NAME, data=np.array(ep_reward_list))




