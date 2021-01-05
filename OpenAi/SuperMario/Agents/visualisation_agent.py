

import matplotlib.pyplot as plt

import Clean_Results.Agents.storage_agent as storage_agent


def show_env_props(env):

    env_name = env.unwrapped.spec.id
    num_states = env.observation_space.shape
    num_actions = env.action_space.n


    print("Env Name ->  {}".format(env_name))

    print("Size of State Space ->  {}".format(num_states))
    print("Size of Action Space ->  {}".format(num_actions))

def plot_episodic_rewards(EPISODES_NAME, REWARDS_NAME, LABEL):

    #EPISODES_NAME = "env-{}__v-{}__ep-{}__stats-{}__episodes".format(env_name, training_version_name, episodes, stats_every)
    #REWARDS_NAME = "env-{}__v-{}__ep-{}__stats-{}__rewards".format(env_name, training_version_name, episodes,stats_every)

    data_ep = storage_agent.load_np(EPISODES_NAME)
    data_avg = storage_agent.load_np(REWARDS_NAME)
    plt.plot(data_ep, data_avg, label=LABEL)

    plt.legend(loc=1)
    plt.show()