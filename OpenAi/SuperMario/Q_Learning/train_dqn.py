

import time
import numpy as np

#from nes_py.wrappers import BinarySpaceToDiscreteSpaceEnv
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT, RIGHT_ONLY
from nes_py.wrappers import JoypadSpace

from OpenAi.SuperMario.Agents.Agent import DQN_Agent
from OpenAi.SuperMario.Agents.wrapper import wrapper
import Clean_Results.Agents.storage_agent as storage_agent




# Parameters

env_name = 'SuperMarioBros-1-1-v0'
training_version_name = 'Deep Q-Learning' # Double
movement_type = 'SIMPLE_MOVEMENT'


states = (84, 84, 4)        # 0-width, 1-height, 2-skip (learn each + 1)
num_hidden = 128
show_model = True

episodes = 1000             # 10000
show_every = 5              # episodes
stats_every = 2             # episodes
save_weights_every = 10     # episodes
save_rewards_every = 10     # episodes

epsilon = 1
eps_decay = 0.99999         # 0.99999975
eps_min = 0.1
gamma = 0.90
double_q = False

max_memory = 100000         # steps
copy = 1000                 # Target ntwork sync - 10000
learn_each = 3              # steps
save_each = 100000          # steps - (500000)
batch_size = 32             # steps


# Params visualisation:

print("_ENVIRONMENT_")
print("env_name: ", env_name)
print("training_version_name: ", training_version_name)
print("movement_type: ", movement_type)
print()
print("learn_each: ", learn_each, " steps")
print("max_memory: ", max_memory, " steps")
print("batch_size: ", batch_size, " steps")
print("sync target each: ", copy, " steps")
print("gamma: ", gamma)
print()


env = gym_super_mario_bros.make(env_name)
env = JoypadSpace(env, SIMPLE_MOVEMENT)                     # env = BinarySpaceToDiscreteSpaceEnv(env, RIGHT_ONLY)
env = wrapper(env, states)

actions = env.action_space.n


agent = DQN_Agent(env=env, states=states, num_hidden=num_hidden, epsilon=epsilon, eps_decay=eps_decay, eps_min=eps_min,
                  max_memory=max_memory, copy=copy, learn_each=learn_each, save_each=save_each, batch_size=batch_size,
                  gamma = gamma, show_model=show_model, double_q=double_q)


# Variables
rewards = []
x_positions = []
stats_ep_rewards = {'ep': [], 'avg': [], 'x_pos': []}
step = 0
frames_avg = 0
# Timing
start = time.time()


# Main loop
for e in range(episodes):

    state = env.reset()

    total_reward = 0
    iter = 0

    while True:
        # env.render()

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


        if info['flag_get']:
            print("Got the Flag! :D")
            break

        # If done break loop
        if done:
            break

        # Rewards
        rewards.append(total_reward / iter)
        if(e == 0): x_positions.append(info['x_pos'])
        else: x_positions.append((np.mean(x_positions) + info['x_pos']) / 2)

    frames_avg = (frames_avg + agent.step) / 2

    # Visualisation
    if e % show_every == 0 and e != 0:
        print('Episode {e} - '
              'Frame {f}/{ft} - '
              'Frames/sec {fs} - '
              'Time {t}/{tt} - '
              'Epsilon {eps} - '
              'Mean Reward {r} - '
              'Mean X position {xp}'.format(e=e,
                                        f=agent.step,
                                        ft=np.round(episodes*frames_avg),
                                        fs=np.round((agent.step - step) / (time.time() - start)),
                                        t=np.round(time.time()-start),
                                        tt=int(np.round((episodes * frames_avg) / ((agent.step - step) / (time.time() - start)))),
                                        eps=np.round(agent.epsilon, 4),
                                        r=np.round(np.mean(rewards[-20:]), 4),    # [-100:]
                                        xp=info['x_pos']))
        start = time.time()
        step = agent.step

    # SAVE WEIGHTS
    if (e % save_weights_every == 0 and e != 0):
        model_file_path = "./{}__{}__{}_online.HDF5".format(env_name, training_version_name, movement_type) # ,episode_count
        agent.model_online.save(model_file_path)
        model_target_file_path = "./{}__{}__{}_target.HDF5".format(env_name, training_version_name, movement_type)  # ,episode_count
        agent.model_target.save(model_target_file_path)
        print("weights saved")

    # Log rewards
    if(e % stats_every == 0 and e != 0):
        stats_ep_rewards['ep'].append(e)
        stats_ep_rewards['avg'].append(np.mean(rewards[-20:]))
        stats_ep_rewards['x_pos'].append(info['x_pos'])

    # SAVE REWARDS
    if (e % save_rewards_every == 0 and e != 0):
        EPISODES_NAME = "{}__{}__{}__stats_ep".format(env_name, training_version_name, movement_type)
        REWARDS_NAME = "{}__{}__{}__stats_avg".format(env_name, training_version_name, movement_type)
        X_POS_NAME = "{}__{}__{}__stats_x_pos".format(env_name, training_version_name, movement_type)
        storage_agent.save_np(name=EPISODES_NAME, data=np.array(stats_ep_rewards['ep']), visualise=False)
        storage_agent.save_np(name=REWARDS_NAME, data=np.array(stats_ep_rewards['avg']), visualise=False)
        storage_agent.save_np(name=X_POS_NAME, data=np.array(stats_ep_rewards['x_pos']), visualise=False)
