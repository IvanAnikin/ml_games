
import OpenAi.MountainCar.Q_Learning.storage_agent as MountainCar_Q_Learning_storage_agent
import matplotlib.pyplot as plt


def visualise_stats(LEARNING_RATES = [0.05, 0.10, 0.15, 0.20], EPSILONS = [0.5], END_EPSILON_DECAYING_POSITIONS = [2.0], DISCOUNTS = [0.95], DISCRETE_OS_SIZES = [20], show_every = 4000, episodes = 5000, stats_every = 100):

    EPISODES_NAME = "ep-{}__stats-{}__episodes".format(episodes, stats_every)
    data_ep = MountainCar_Q_Learning_storage_agent.load_np(EPISODES_NAME)
    for learning_rate_cycle in range(len(LEARNING_RATES)):
        for epsilon_cycle in range(len(EPSILONS)):
            for end_epsilon_decaying_cycle in range(len(END_EPSILON_DECAYING_POSITIONS)):
                for discount_cycle in range(len(DISCOUNTS)):
                    for discrete_os_size_cycle in range(len(DISCRETE_OS_SIZES)):
                        
                        learning_rate = LEARNING_RATES[learning_rate_cycle]
                        epsilon = EPSILONS[epsilon_cycle]
                        end_epsilon_decaying = END_EPSILON_DECAYING_POSITIONS[end_epsilon_decaying_cycle]
                        discount = DISCOUNTS[discount_cycle]
                        discrete_os_size = [DISCRETE_OS_SIZES[discrete_os_size_cycle], DISCRETE_OS_SIZES[discrete_os_size_cycle]]

                        #NAME = "ep-{}__stats-{}__lr-{}".format(episodes, stats_every, learning_rate)
                        NAME = "ep-{}__stats-{}__lr-{}__eps-{}__epsDec-{}__disc-{}__size-{}".format(episodes, stats_every, learning_rate, epsilon, end_epsilon_decaying, discount, discrete_os_size)

                        data_avg = MountainCar_Q_Learning_storage_agent.load_np(NAME)
                        plt.plot(data_ep, data_avg, label=NAME)
                        # stats_avg.append(MountainCar_Q_Learning_storage_agent.load_np(NAME))

    plt.legend(loc=1)
    plt.show()


