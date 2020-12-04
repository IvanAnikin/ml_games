
import OpenAi.MountainCar.Q_Learning.gym_agent as MountainCar_Q_Learning_gym_agent
import OpenAi.MountainCar.Q_Learning.storage_agent as MountainCar_Q_Learning_storage_agent
import OpenAi.MountainCar.Q_Learning.visualisation_agent as MountainCar_Q_Learning_visualisation_agent
import OpenAi.MountainCar.Q_Learning.stats_agent as MountainCar_Q_Learning_stats_agent

import matplotlib.pyplot as plt
import numpy as np





#LEARNING_RATES = [0.15, 0.20]
#EPSILONS = [0.2, 0.5, 0.7]
#END_EPSILON_DECAYING_POSITIONS = [1.5, 2.0, 2.5]
#DISCOUNTS = [0.7, 0.85, 0.95]
#DISCRETE_OS_SIZES = [10, 20, 30]

#episodes = 10000
#show_every = 1000
#stats_every = 200

                                     
                                            
LEARNING_RATES = [0.15]
EPSILONS = [0.7]
END_EPSILON_DECAYING_POSITIONS = [1.5]
DISCOUNTS = [0.95]
DISCRETE_OS_SIZES = [20]

episodes = 30000
show_every = 2000
stats_every = 40000



#MountainCar_Q_Learning_gym_agent.save_games(LEARNING_RATES=LEARNING_RATES, EPSILONS= EPSILONS, END_EPSILON_DECAYING_POSITIONS=END_EPSILON_DECAYING_POSITIONS, DISCOUNTS=DISCOUNTS, DISCRETE_OS_SIZES=DISCRETE_OS_SIZES, episodes=episodes, show_every=show_every, stats_every=stats_every)

#MountainCar_Q_Learning_visualisation_agent.visualise_stats(LEARNING_RATES=LEARNING_RATES, EPSILONS= EPSILONS, END_EPSILON_DECAYING_POSITIONS=END_EPSILON_DECAYING_POSITIONS, DISCOUNTS=DISCOUNTS, DISCRETE_OS_SIZES=DISCRETE_OS_SIZES, episodes=episodes, show_every=show_every, stats_every=stats_every)

#print(MountainCar_Q_Learning_stats_agent.best_rewards_params(score_count_start_position=9000, LEARNING_RATES=LEARNING_RATES, EPSILONS= EPSILONS, END_EPSILON_DECAYING_POSITIONS=END_EPSILON_DECAYING_POSITIONS, DISCOUNTS=DISCOUNTS, DISCRETE_OS_SIZES=DISCRETE_OS_SIZES, episodes=episodes, show_every=show_every, stats_every=stats_every))



#_, q_table = MountainCar_Q_Learning_gym_agent.mountain_car_single_game(EPISODES=episodes, LEARNING_RATE=LEARNING_RATES[0], epsilon=EPSILONS[0], end_epsilon_decaying_position=END_EPSILON_DECAYING_POSITIONS[0], DISCOUNT=DISCOUNTS[0], DISCRETE_OS_SIZE=[DISCRETE_OS_SIZES[0], DISCRETE_OS_SIZES[0]], SHOW_EVERY=show_every, STATS_EVERY=stats_every)

#MountainCar_Q_Learning_storage_agent.save_np("q_table_30000__lr-0.2__eps-0.7__epsDec-1.5__disc-0.95__size-[20, 20]", q_table)

q_table = MountainCar_Q_Learning_storage_agent.load_np("q_table_30000__lr-0.2__eps-0.7__epsDec-1.5__disc-0.95__size-[20, 20]")
successful_episodes, average_score = MountainCar_Q_Learning_gym_agent.play_with_given_q_table(q_table, [20, 20], 2000, 1)
print("successful_episodes: ", successful_episodes)
print("average_score: ", average_score)




