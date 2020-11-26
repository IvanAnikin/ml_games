
import OpenAi.MountainCar.Q_Learning.gym_agent as MountainCar_Q_Learning_gym_agent
import OpenAi.MountainCar.Q_Learning.storage_agent as MountainCar_Q_Learning_storage_agent
import OpenAi.MountainCar.Q_Learning.visualisation_agent as MountainCar_Q_Learning_visualisation_agent

import matplotlib.pyplot as plt
import numpy as np





#LEARNING_RATES = [0.15, 0.20]
#EPSILONS = [0.2, 0.5, 0.7]
#END_EPSILON_DECAYING_POSITIONS = [1.5, 2.0, 2.5]
#DISCOUNTS = [0.70, 0.85, 0.95]
#DISCRETE_OS_SIZES = [10, 20, 30]

#episodes = 5000
#show_every = 1000
#stats_every = 100


LEARNING_RATES = [0.15]
EPSILONS = [0.5, 0.7]
END_EPSILON_DECAYING_POSITIONS = [2.0]
DISCOUNTS = [0.95]
DISCRETE_OS_SIZES = [10, 20]

episodes = 5000
show_every = 1000
stats_every = 100



MountainCar_Q_Learning_gym_agent.save_games(LEARNING_RATES=LEARNING_RATES, EPSILONS= EPSILONS, END_EPSILON_DECAYING_POSITIONS=END_EPSILON_DECAYING_POSITIONS, DISCOUNTS=DISCOUNTS, DISCRETE_OS_SIZES=DISCRETE_OS_SIZES, episodes=episodes, show_every=show_every, stats_every=stats_every)

#MountainCar_Q_Learning_visualisation_agent.visualise_stats(LEARNING_RATES=LEARNING_RATES, EPSILONS= EPSILONS, END_EPSILON_DECAYING_POSITIONS=END_EPSILON_DECAYING_POSITIONS, DISCOUNTS=DISCOUNTS, DISCRETE_OS_SIZES=DISCRETE_OS_SIZES, episodes=episodes, show_every=show_every, stats_every=stats_every)

