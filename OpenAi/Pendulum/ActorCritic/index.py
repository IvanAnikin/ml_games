
import gym
import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt

import OpenAi.Pendulum.ActorCritic.visualisation as visualiser
import OpenAi.Pendulum.ActorCritic.Agent as Agent_file
import OpenAi.MountainCar.Q_Learning.storage_agent as MountainCar_Q_Learning_storage_agent




std_dev = 0.2

gamma = 0.99 # Discount factor for future rewards
tau = 0.005 # Used to update target networks

buffer_capacity=50000
batch_size=64

critic_lr = 0.002
actor_lr = 0.001

episodes = 100
stats_every = 40


env_name = "Pendulum-v0"
env = gym.make(env_name)

visualiser.show_env_props(env)


ou_noise = Agent_file.OUActionNoise(mean=np.zeros(1), std_deviation=float(std_dev) * np.ones(1))
Agent = Agent_file.Agent(env=env, buffer_capacity=buffer_capacity, batch_size=batch_size, gamma=gamma, tau=tau, critic_lr = critic_lr, actor_lr = actor_lr)



ep_reward_list = [] # To store average reward history of last few episodes
avg_reward_list = []

for ep in range(episodes):

    prev_state = env.reset()
    episodic_reward = 0

    while True:

        tf_prev_state = tf.expand_dims(tf.convert_to_tensor(prev_state), 0)

        action = Agent.action(tf_prev_state, ou_noise) # Recieve state and reward from environment.
        state, reward, done, info = env.step(action)

        Agent.record((prev_state, action, reward, state))
        episodic_reward += reward

        Agent.learn()

        if done: # End this episode when `done` is True
            break

        prev_state = state

    ep_reward_list.append(episodic_reward)

    # Mean of last 40 episodes
    avg_reward = np.mean(ep_reward_list[-stats_every:])
    print("Episode * {} * Avg Reward is ==> {}".format(ep, avg_reward))
    avg_reward_list.append(avg_reward)


plt.plot(avg_reward_list)
plt.xlabel("Episode")
plt.ylabel("Avg. Epsiodic Reward")
plt.show()

NAME = "{}_ep-{}__stats-{}__lr-{}__eps-{}__epsDec-{}__disc-{}__size-{}".format(env_name, episodes, stats_every, gamma,
                                                                            tau, buffer_capacity, batch_size,
                                                                            critic_lr, actor_lr)
MountainCar_Q_Learning_storage_agent.save_np(name=NAME, data=np.array(avg_reward_list))

