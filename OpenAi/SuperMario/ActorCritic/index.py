
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





env_name = "SuperMarioBros-v0"
training_version_name = "Actor_Critic_1"
movement_type = "SIMPLE_MOVEMENT"
env = gym_super_mario_bros.make(env_name)
env = JoypadSpace(env, SIMPLE_MOVEMENT)


num_hidden_units = 128
# Discount factor for future rewards
gamma = 0.99

max_episodes = 10000
max_steps_per_episode = 1000

# Cartpole-v0 is considered solved if average reward is >= 195 over 100
# consecutive trials
reward_threshold = 195
running_reward = 0

agent = Agent_file.Actor_Critic_1(env=env, gamma=gamma, num_hidden_units=num_hidden_units)




with tqdm.trange(max_episodes) as t:
  for i in t:
    initial_state = tf.constant(env.reset(), dtype=tf.float32)
    episode_reward = int(agent.train_step(initial_state, max_steps_per_episode))

    running_reward = episode_reward*0.01 + running_reward*.99

    t.set_description(f'Episode {i}')
    t.set_postfix(episode_reward=episode_reward, running_reward=running_reward)

    # Show average episode reward every n episodes - !


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

#storage_agent.save_np(name=EPISODES_NAME, data=np.array(ep_stats))
#storage_agent.save_np(name=REWARDS_NAME, data=np.array(ep_reward_list))




