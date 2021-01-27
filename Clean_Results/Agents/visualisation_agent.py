
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

def plot_data(X_name, Y_name, NAME):
    data_avg = storage_agent.load_np(X_name)
    data_ep = storage_agent.load_np(Y_name)
    plt.plot(data_ep, data_avg, label=NAME)

def find_max(name):
    data = storage_agent.load_np(name)
    max_record = 0
    index = 0

    for record in data:
        if(record > max_record): max_record = record
        index+=1

    return max_record, index