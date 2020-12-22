
import gym
import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt

import OpenAi.Pendulum.ActorCritic.visualisation as visualiser
import OpenAi.Pendulum.ActorCritic.Agent as Agent_file
import OpenAi.MountainCar.Q_Learning.storage_agent as MountainCar_Q_Learning_storage_agent




std_devs = [0.15, 0.20, 0.25]

gammas = [0.90, 0.95, 0.99] # Discount factor for future rewards
taus = [0.007, 0.005, 0.001] # Used to update target networks

buffer_capacitys =[30000, 50000, 100000]
batch_sizes =[32, 64, 128]

critic_lrs = [0.0015, 0.0020, 0.0025]
actor_lrs = [0.0005, 0.001, 0.0015]

episodes = 100
stats_every = 40


env_name = "Pendulum-v0"
env = gym.make(env_name)

visualiser.show_env_props(env)

for std_devc in range(len(std_devs)):
    for gammac in range(len(gammas)):
        for tauc in range(len(taus)):
            for buffer_capacityc in range(len(buffer_capacitys)):
                for batch_sizec in range(len(batch_sizes)):
                    for critic_lrc in range(len(critic_lrs)):
                        for actor_lrc in range(len(actor_lrs)):

                            std_dev = std_devs[std_devc]
                            gamma = gammas[gammac]
                            tau = taus[tauc]
                            buffer_capacity = buffer_capacitys[buffer_capacityc]
                            batch_size = batch_sizes[batch_sizec]
                            critic_lr = critic_lrs[critic_lrc]
                            actor_lr = actor_lrs[actor_lrc]

                            NAME = "{}__episodes-{}__stats_every-{}__std_devc-{}__gamma-{}__tau-{}__buffer_capacity-{}__batch_size-{}__critic_lr-{}__actor_lr-{}".format(env_name,
                                                                                                           episodes,
                                                                                                           stats_every,
                                                                                                           std_devc,
                                                                                                           gamma,
                                                                                                           tau,
                                                                                                           buffer_capacity,
                                                                                                           batch_size,
                                                                                                           critic_lr,
                                                                                                            actor_lr)
                            '''   
                            Weights_Acor_Name = "Weights_Acor___" + NAME + ".h5"
                            Weights_Critic_Name = "Weights_Critic___" + NAME + ".h5"
                            Weights_Acor_Target_Name = "Weights_Acor_Target___" + NAME + ".h5"
                            Weights_Critic_Target_Name = "Weights_Critic_Target___" + NAME + ".h5"
                            '''

                            print(NAME)

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



                            MountainCar_Q_Learning_storage_agent.save_np(name=NAME, data=np.array(avg_reward_list))

