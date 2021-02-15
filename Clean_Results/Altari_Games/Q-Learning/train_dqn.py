

import time
import numpy as np
from PIL import Image

import gym
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT, RIGHT_ONLY
from nes_py.wrappers import JoypadSpace

from Clean_Results.Agents.wrapper import wrapper
from Clean_Results.Agents.Agent import DQN_Agent
import Clean_Results.Agents.storage_agent as storage_agent





# done:
# SpaceInvaders-v0

# todo:
# BankHeist-v0
# BeamRider-v0
# Bowling-v0
# Breakout-v0

# double deep SpaceInvaders train 10000 ep

env_name = 'AirRaid-v0'
training_version_name = 'Deep Q-Learning' # Double
movement_type = "Default"


states = (84, 84, 4)
num_hidden = 128
show_model = True
load_model = True
model_file_path_online = "./{}__{}__{}_online.HDF5".format(env_name, training_version_name, movement_type)
model_file_path_target = "./{}__{}__{}_target.HDF5".format(env_name, training_version_name, movement_type)

episodes = 5000             # 10000
show_every = 5              # episodes
stats_every = 2             # episodes
save_weights_every = 10     # episodes
save_rewards_every = 10     # episodes

epsilon = 1                 # 1
eps_decay = 0.999999        # avg 0.4 epsilon left after 300 episodes (0.99999975)
eps_min = 0.1
gamma = 0.90
double_q = False

max_memory = 10000          # steps 100000
copy = 1000                 # Target ntwork sync - 10000
learn_each = 3              # steps
save_each = 100000          # steps - (500000) #NOT USED
batch_size = 32             # steps


# Params visualisation:

print("_ENVIRONMENT_")
print("env_name: ", env_name)
print("training_version_name: ", training_version_name)
#print("movement_type: ", movement_type)
print()
print("learn_each: ", learn_each, " steps")
print("max_memory: ", max_memory, " steps")
print("batch_size: ", batch_size, " steps")
print("sync target each: ", copy, " steps")
print()
print("epsilon: ", epsilon)
print("eps_decay: ", eps_decay)
print("eps_min: ", eps_min)
print("gamma: ", gamma)
print()


env = gym.make(env_name)
#env = JoypadSpace(env, SIMPLE_MOVEMENT)                     # env = BinarySpaceToDiscreteSpaceEnv(env, RIGHT_ONLY)
env = wrapper(env, states)

actions = env.action_space.n

agent = DQN_Agent(env=env, states=states, num_hidden=num_hidden, epsilon=epsilon, eps_decay=eps_decay, eps_min=eps_min,
                  max_memory=max_memory, copy=copy, learn_each=learn_each, save_each=save_each, batch_size=batch_size,
                  gamma = gamma, show_model=show_model, load_model=load_model, model_file_path_online=model_file_path_online,
                  model_file_path_target=model_file_path_target, double_q=double_q)


# Variables
rewards = []
steps = []
frames_stats = []
stats_ep_rewards = {'ep': [], 'avg': [], 'steps': [], 'epsilon': []}
step = 0
frames_avg = 0
# Timing
start = time.time()
start_ep = time.time()
max_ep_steps = 0


# Main loop
for e in range(episodes):

    state = env.reset()

    total_reward = 0
    iter = 0
    frames = []

    while True:
        # env.render()
        frames.append(Image.fromarray(env.render(mode='rgb_array')))

        # Run agent
        action = agent.run(state=state)

        # Perform action
        next_state, reward, done, info = env.step(action=action)

        # Remember transition
        agent.add(experience=(state, next_state, action, reward, done))

        # Update agent
        agent.learn()

        # Total reward
        total_reward += reward

        # Update state
        state = next_state

        # Increment
        iter += 1

        # If done break loop
        if done:
            done = False
            break

    # Rewards
    rewards.append(total_reward / iter)
    frames_stats.append(agent.step)

    ep_steps = agent.step - step

    steps.append(ep_steps)

    if (ep_steps > max_ep_steps):
        max_ep_steps = ep_steps
        with open('max_ep_steps' + env_name + '.gif', 'wb') as f:
            im = Image.new('RGB', frames[0].size)
            im.save(f, save_all=True, append_images=frames)

    # Visualisation
    if e % show_every == 0 and e != 0:
        print('Episode {e} - '
              'Frame {f}/{ft} - '
              'Frames/sec {fs} - '
              'Time {t_ep} | {t}/{tt} - '
              'Epsilon {eps} - '
              'Mean Reward {r} - '
              'Mean steps {s} - '
              'Max steps {sm}'.format(e=e,
                                            f=agent.step,
                                            ft=int(np.round(episodes * np.mean(frames_stats))),
                                            fs=np.round((agent.step - step) / (time.time() - start_ep)),
                                            t=int(np.round(time.time() - start)),
                                            t_ep=int(np.round(time.time() - start_ep)),
                                            tt=int(np.round((episodes * np.mean(frames_stats)) / (
                                                            (agent.step - step) / (time.time() - start)))),
                                            eps=np.round(agent.epsilon, 4),
                                            r=np.round(np.mean(rewards[-20:]), 4),  # [-100:]
                                            s=int(np.round(np.mean(steps[-20:]))),
                                            sm=max_ep_steps)
              )
    start_ep = time.time()
    step = agent.step

    # SAVE WEIGHTS
    if (e % save_weights_every == 0 and e != 0):
        model_file_path = "./{}__{}__{}_online.HDF5".format(env_name, training_version_name,
                                                            movement_type)  # ,episode_count
        agent.model_online.save(model_file_path)
        model_target_file_path = "./{}__{}__{}_target.HDF5".format(env_name, training_version_name,
                                                                   movement_type)  # ,episode_count
        agent.model_target.save(model_target_file_path)
        print("weights saved")

    # Log rewards
    if (e % stats_every == 0 and e != 0):
        stats_ep_rewards['ep'].append(e)
        stats_ep_rewards['avg'].append(np.mean(rewards[-20:]))
        stats_ep_rewards['steps'].append(np.mean(steps[-20:]))
        stats_ep_rewards['epsilon'].append(agent.epsilon)

    # SAVE REWARDS
    if (e % save_rewards_every == 0 and e != 0):
        EPISODES_NAME = "{}__{}__{}__stats_ep".format(env_name, training_version_name, movement_type)
        REWARDS_NAME = "{}__{}__{}__stats_avg".format(env_name, training_version_name, movement_type)
        STEPS_NAME = "{}__{}__{}__stats_steps".format(env_name, training_version_name, movement_type)
        EPSILON_NAME = "{}__{}__{}__stats_steps".format(env_name, training_version_name, movement_type)
        storage_agent.save_np(name=EPISODES_NAME, data=np.array(stats_ep_rewards['ep']), visualise=False)
        storage_agent.save_np(name=REWARDS_NAME, data=np.array(stats_ep_rewards['avg']), visualise=False)
        storage_agent.save_np(name=STEPS_NAME, data=np.array(stats_ep_rewards['steps']), visualise=False)
        storage_agent.save_np(name=EPSILON_NAME, data=np.array(stats_ep_rewards['epsilon']), visualise=False)

