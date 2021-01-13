
import matplotlib.pyplot as plt
# gifs displaying
from JSAnimation.IPython_display import display_animation
from matplotlib import animation
from IPython.display import display

import Clean_Results.Agents.storage_agent as storage_agent


def show_env_props(env):

    env_name = env.unwrapped.spec.id
    num_states = env.observation_space.shape
    num_actions = env.action_space.n


    print("Env Name ->  {}".format(env_name))

    print("Size of State Space ->  {}".format(num_states))
    print("Size of Action Space ->  {}".format(num_actions))

def plot_episodic_rewards(EPISODES_NAME, REWARDS_NAME, LABEL):

    data_ep = storage_agent.load_np(EPISODES_NAME)
    data_avg = storage_agent.load_np(REWARDS_NAME)
    plt.plot(data_ep, data_avg, label=LABEL)

    plt.legend(loc=1)
    plt.show()

def plot_episodic_rewards_with_params(env_name, training_version_name, movement_type, episodes, LABEL):

    EPISODES_NAME = "{}__{}__{}__stats_ep__{}".format(env_name, training_version_name, movement_type, episodes)
    REWARDS_NAME = "{}__{}__{}__stats_avg__{}".format(env_name, training_version_name, movement_type, episodes)

    data_ep = storage_agent.load_np(EPISODES_NAME)
    data_avg = storage_agent.load_np(REWARDS_NAME)

    plt.plot(data_ep, data_avg, label=LABEL)

    plt.legend(loc=1)
    plt.show()

def plot_episodic_rewards_with_params_x_pos(env_name, training_version_name, movement_type, episodes, LABEL):

    EPISODES_NAME = "{}__{}__{}__stats_ep__{}".format(env_name, training_version_name, movement_type, episodes)
    X_POS_NAME = "{}__{}__{}__stats_x_pos__{}".format(env_name, training_version_name, movement_type, episodes)

    data_ep = storage_agent.load_np(EPISODES_NAME)
    data_x_pos = storage_agent.load_np(X_POS_NAME)

    plt.plot(data_ep, data_x_pos, label=LABEL)

    plt.legend(loc=1)
    plt.show()




# not working

def display_frames_as_gif(frames):
    """
    Displays a list of frames as a gif, with controls
    """
    # plt.figure(figsize=(frames[0].shape[1] / 72.0, frames[0].shape[0] / 72.0), dpi = 72)
    patch = plt.imshow(frames[0])
    plt.axis('off')

    def animate(i):
        patch.set_data(frames[i])

    anim = animation.FuncAnimation(plt.gcf(), animate, frames=len(frames), interval=50)
    display(display_animation(anim, default_mode='loop'))