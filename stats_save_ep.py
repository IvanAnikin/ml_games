import OpenAi.MountainCar.Q_Learning.gym_agent as MountainCar_Q_Learning_gym_agent
import OpenAi.MountainCar.Q_Learning.storage_agent as MountainCar_Q_Learning_storage_agent
import OpenAi.MountainCar.Q_Learning.visualisation_agent as MountainCar_Q_Learning_visualisation_agent

import matplotlib.pyplot as plt
import numpy as np




LEARNING_RATES = [0.15, 0.20]
EPSILONS = [0.2, 0.5, 0.7]
END_EPSILON_DECAYING_POSITIONS = [1.5, 2.0, 2.5]
DISCOUNTS = [0.70, 0.85, 0.95]
DISCRETE_OS_SIZES = [10, 20, 30]

episodes = 10000
show_every = 1000
stats_every = 200

round = 0
stats_ep_rewards = []

						
for episode in range(episodes):
	if not episode % stats_every: 
		stats_ep_rewards.append(episode) 
	 
						
print(stats_ep_rewards[10])
EPISODES_NAME = "ep-{}__stats-{}__episodes".format(episodes, stats_every)
MountainCar_Q_Learning_storage_agent.save_np(name=EPISODES_NAME, data=np.array(stats_ep_rewards))