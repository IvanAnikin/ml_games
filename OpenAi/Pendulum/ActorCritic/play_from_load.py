
import gym
import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt

import OpenAi.Pendulum.ActorCritic.visualisation as visualiser
import OpenAi.Pendulum.ActorCritic.Agent as Agent_file
import OpenAi.MountainCar.Q_Learning.storage_agent as MountainCar_Q_Learning_storage_agent
import OpenAi.Agents.storage_agent as storage_agent




std_dev = 0.2

gamma = 0.99 # Discount factor for future rewards
tau = 0.005 # Used to update target networks

buffer_capacity=50000
batch_size=64

critic_lr = 0.002
actor_lr = 0.001

episodes = 5
stats_every = 10


env_name = "Pendulum-v0"
env = gym.make(env_name)

visualiser.show_env_props(env)


ou_noise = Agent_file.OUActionNoise(mean=np.zeros(1), std_deviation=float(std_dev) * np.ones(1))
Agent = Agent_file.Agent(env=env, buffer_capacity=buffer_capacity, batch_size=batch_size, gamma=gamma, tau=tau, critic_lr = critic_lr, actor_lr = actor_lr)



ep_reward_list = [] # To store average reward history of last few episodes
avg_reward_list = []

weights_ep = 25
Agent.models.actor_model.load_weights("pendulum_actor__{}.h5".format(weights_ep))
Agent.models.critic_model.load_weights("pendulum_critic__{}.h5".format(weights_ep))

Agent.models.target_actor.load_weights("pendulum_target_actor__{}.h5".format(weights_ep))
Agent.models.target_critic.load_weights("pendulum_target_critic__{}.h5".format(weights_ep))

for ep in range(episodes):

    prev_state = env.reset()
    episodic_reward = 0

    frames = []
    while True:
        env.render()
        frames.append(env.render(mode="rgb_array"))

        tf_prev_state = tf.expand_dims(tf.convert_to_tensor(prev_state), 0)

        #action = Agent.action(tf_prev_state, ou_noise) # Recieve state and reward from environment.
        action = env.action_space.sample()

        state, reward, done, info = env.step(action)

        episodic_reward += reward

        if done: # End this episode when `done` is True
            break

        prev_state = state

    ep_reward_list.append(episodic_reward)

    name = "{}__gif__{}-th_step__{}".format(env_name, weights_ep, ep)
    storage_agent.save_frames_as_gif(frames=frames, name=name)

    # Mean of last 40 episodes
    avg_reward = np.mean(ep_reward_list[-stats_every:])
    print("Episode * {} * Avg Reward is ==> {}".format(ep, avg_reward))
    avg_reward_list.append(avg_reward)

