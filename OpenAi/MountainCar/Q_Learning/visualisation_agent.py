
import OpenAi.MountainCar.Q_Learning.storage_agent as MountainCar_Q_Learning_storage_agent
import matplotlib.pyplot as plt


def visualise_stats(LEARNING_RATES = [0.05, 0.10, 0.15, 0.20], episodes = 5000, stats_every = 100):

    EPISODES_NAME = "ep-{}__stats-{}__episodes".format(episodes, stats_every)
    data_ep = MountainCar_Q_Learning_storage_agent.load_np(EPISODES_NAME)
    for learning_rate_cycle in range(len(LEARNING_RATES)):
        learning_rate = LEARNING_RATES[learning_rate_cycle]
        NAME = "ep-{}__stats-{}__lr-{}".format(episodes, stats_every, learning_rate)
        data_avg = MountainCar_Q_Learning_storage_agent.load_np(NAME)
        plt.plot(data_ep, data_avg, label=NAME)
        # stats_avg.append(MountainCar_Q_Learning_storage_agent.load_np(NAME))

    plt.legend(loc=1)
    plt.show()


