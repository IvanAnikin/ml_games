
import numpy as np
import matplotlib.pyplot as plt

import Clean_Results.Agents.storage_agent as storage_agent



def show_env_props(env):
    env_name = env.unwrapped.spec.id
    num_states = env.observation_space.shape[0]
    num_actions = env.action_space.shape[0]

    upper_bound = env.action_space.high[0]
    lower_bound = env.action_space.low[0]


    print("Env Name ->  {}".format(env_name))

    print("Size of State Space ->  {}".format(num_states))
    print("Size of Action Space ->  {}".format(num_actions))

    print("Max Value of Action ->  {}".format(upper_bound))
    print("Min Value of Action ->  {}".format(lower_bound))

def plot_data(X_name, Y_name, xlabel, ylabel, NAME):
    data_avg = storage_agent.load_np(X_name)
    data_ep = storage_agent.load_np(Y_name)


    plt.plot(data_ep, data_avg, label=NAME)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(NAME)
    plt.show()

def plot_mario_data_x_pos():
    env_name = 'SuperMarioBros-1-1-v0'
    training_version_name = 'Deep Q-Learning'  # Double
    movement_type = 'SIMPLE_MOVEMENT'

    EPISODES_NAME = "{}__{}__{}__stats_ep".format(env_name, training_version_name, movement_type)
    X_POS_NAME = "{}__{}__{}__stats_x_pos".format(env_name, training_version_name, movement_type)
    NAME = "SuperMario DQN"
    xlabel = "Episode"
    ylabel = "Avg. distance"

    plot_data(X_POS_NAME, EPISODES_NAME, xlabel, ylabel, NAME)

def plot_mario_data_rewards():
    env_name = 'SuperMarioBros-1-1-v0'
    training_version_name = 'Deep Q-Learning'  # Double
    movement_type = 'SIMPLE_MOVEMENT'

    EPISODES_NAME = "{}__{}__{}__stats_ep".format(env_name, training_version_name, movement_type)
    X_POS_NAME = "{}__{}__{}__stats_avg".format(env_name, training_version_name, movement_type)
    NAME = "SuperMario DQN"
    xlabel = "Episode"
    ylabel = "Avg. reward"

    plot_data(X_POS_NAME, EPISODES_NAME, xlabel, ylabel, NAME)

def plot_altari_data_rewards():
    env_name = 'SpaceInvaders-v0'
    training_version_name = 'Deep Q-Learning'  # Double
    movement_type = "Default"

    EPISODES_NAME = "{}__{}__{}__stats_ep".format(env_name, training_version_name, movement_type)
    X_POS_NAME = "{}__{}__{}__stats_avg".format(env_name, training_version_name, movement_type)
    NAME = "SpaceInvaders DQN"
    xlabel = "Episode"
    ylabel = "Avg. reward"

    env_name = 'SpaceInvaders-v0'
    training_version_name = 'Deep Q-Learning'  # Double
    movement_type = "Default"

    EPISODES_NAME = "{}__{}__{}__stats_ep".format(env_name, training_version_name, movement_type)
    REWARDS_NAME = "{}__{}__{}__stats_avg".format(env_name, training_version_name, movement_type)

    data = storage_agent.load_np(REWARDS_NAME)
    data_ep = storage_agent.load_np(EPISODES_NAME)

    data2 = []
    data3 = np.zeros(shape=(4995,))
    index = 0
    for entry in data:

        if (index < 9990): data3[int(index / 2)] = data[index] * 10

        index += 2

    plt.plot(data_ep, data3, label=NAME)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(NAME)
    plt.show()

def plot_altari_data_steps():
    env_name = 'SpaceInvaders-v0'
    training_version_name = 'Deep Q-Learning'  # Double
    movement_type = "Default"

    EPISODES_NAME = "{}__{}__{}__stats_ep".format(env_name, training_version_name, movement_type)
    X_POS_NAME = "{}__{}__{}__stats_steps".format(env_name, training_version_name, movement_type)
    NAME = "SpaceInvaders DQN"
    xlabel = "Episode"
    ylabel = "Epsilon"
    #ylabel = "Avg. episode steps"

    plot_data(X_POS_NAME, EPISODES_NAME, xlabel, ylabel, NAME)

def plot_airrapid_data_steps():
    env_name = 'SpaceInvaders-v0'
    training_version_name = 'Deep Q-Learning'  # Double
    movement_type = "Default"

    EPISODES_NAME = "{}__{}__{}__stats_ep".format(env_name, training_version_name, movement_type)
    X_POS_NAME = "{}__{}__{}__stats_steps".format(env_name, training_version_name, movement_type)
    NAME = "SpaceInvaders DQN"
    xlabel = "Episode"
    ylabel = "Epsilon"
    #ylabel = "Avg. episode steps"

    plot_data(X_POS_NAME, EPISODES_NAME, xlabel, ylabel, NAME)

def plot_altari_data_epsilon():
    env_name = 'SpaceInvaders-v0'
    training_version_name = 'Deep Q-Learning'  # Double
    movement_type = "Default"

    EPISODES_NAME = "{}__{}__{}__stats_ep".format(env_name, training_version_name, movement_type)
    X_POS_NAME = "{}__{}__{}__stats_epsilon".format(env_name, training_version_name, movement_type)
    NAME = "SpaceInvaders DQN"
    xlabel = "Episode"
    ylabel = "Epsilon"

    plot_data(X_POS_NAME, EPISODES_NAME, xlabel, ylabel, NAME)

def find_max(name):
    data = storage_agent.load_np(name)
    max_record = 0
    index = 0

    for record in data:
        if(record > max_record): max_record = record
        index+=1

    return max_record, index

def convert_altari_rewards():

    env_name = 'SpaceInvaders-v0'
    training_version_name = 'Deep Q-Learning'  # Double
    movement_type = "Default"

    EPISODES_NAME = "{}__{}__{}__stats_ep".format(env_name, training_version_name, movement_type)
    REWARDS_NAME = "{}__{}__{}__stats_avg".format(env_name, training_version_name, movement_type)

    data = storage_agent.load_np(REWARDS_NAME)

    data2 = []
    data3 = np.zeros(shape=(2945,))
    index = 0
    for entry in data:

        if(index < 4990): data3[int(index/2)] = data[index]

        index+=2

    print("converted arary with shape: ", data.shape, " to: ", data3.shape)

    return data2