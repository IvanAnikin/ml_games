

import time
import numpy as np
from PIL import Image

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
load_model = True
model_file_path_online = "./{}__{}__{}_online.HDF5".format(env_name, training_version_name, movement_type)
model_file_path_target = "./{}__{}__{}_target.HDF5".format(env_name, training_version_name, movement_type)

episodes = 1000             # 10000
show_every = 5              # episodes
stats_every = 2             # episodes
save_weights_every = 10     # episodes
save_rewards_every = 10     # episodes

epsilon = 0                 # 1
eps_decay = 0               # avg 0.3 epsilon left after 300 episodes (0.99999975)
eps_min = 0
gamma = 0.90
double_q = True

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
print()
print("epsilon: ", epsilon)
print("eps_decay: ", eps_decay)
print("gamma: ", gamma)
print()


env = gym_super_mario_bros.make(env_name)
env = JoypadSpace(env, SIMPLE_MOVEMENT)                     # env = BinarySpaceToDiscreteSpaceEnv(env, RIGHT_ONLY)
env = wrapper(env, states)

actions = env.action_space.n


agent = DQN_Agent(env=env, states=states, num_hidden=num_hidden, epsilon=epsilon, eps_decay=eps_decay, eps_min=eps_min,
                  max_memory=max_memory, copy=copy, learn_each=learn_each, save_each=save_each, batch_size=batch_size,
                  gamma = gamma, show_model=show_model, load_model=load_model, model_file_path_online=model_file_path_online,
                  model_file_path_target=model_file_path_target, double_q=double_q)


# Variables
rewards = []
x_positions = []
frames_stats = []
stats_ep_rewards = {'ep': [], 'avg': [], 'x_pos': []}
step = 0
frames_avg = 0
# Timing
start = time.time()
start_ep = time.time()
max_x_pos = 0



# Main loop
for e in range(episodes):

    state = env.reset()

    total_reward = 0
    iter = 0
    frames = []

    while True:
        #env.render()
        frames.append(Image.fromarray(env.render(mode='rgb_array')))

        # Run agent
        action = agent.run(state=state)

        # Perform action
        next_state, reward, done, info = env.step(action=action)

        # Remember transition
        agent.add(experience=(state, next_state, action, reward, done))

        # Update agent
        #agent.learn()

        # Total reward
        total_reward += reward

        # Update state
        state = next_state

        # Increment
        iter += 1


        if info['flag_get']:
            print("Got the Flag! :D")
            print("x pos: {} - steps: {} - episode reward: {}".format(info['x_pos'], step, total_reward))
            with open('getting_flag_{}.gif'.format(e), 'wb') as f:  # change the path if necessary
                im = Image.new('RGB', frames[0].size)
                im.save(f, save_all=True, append_images=frames)
            # save with big reward
            break

        # If done break loop
        if done:
            break

    # Rewards
    rewards.append(total_reward / iter)
    frames_stats.append(agent.step)
    if(e == 0): x_positions.append(info['x_pos'])
    else: x_positions.append(info['x_pos'])    #((np.mean(x_positions) + info['x_pos']) / 2)

    if(info['x_pos'] > max_x_pos):
        max_x_pos = info['x_pos']
        with open('max_x_pos.gif', 'wb') as f:  # change the path if necessary
            im = Image.new('RGB', frames[0].size)
            im.save(f, save_all=True, append_images=frames)

    #frames_avg = (frames_avg + agent.step) / 2

    # Visualisation
    if e % show_every == 0 and e != 0:
        print('Episode {e} - '
              'Frame {f}/{ft} - '
              'Frames/sec {fs} - '
              'Time {t_ep} | {t}/{tt} - '
              'Epsilon {eps} - '
              'Mean Reward {r} - '
              'Mean X position {xp} - '
              'Max X position {xm}'.format(e=e,
                                        f=agent.step,
                                        ft=int(np.round(episodes* np.mean(frames_stats))),
                                        fs=np.round((agent.step - step) / (time.time() - start_ep)),
                                        t=int(np.round(time.time()-start)),
                                        t_ep=int(np.round(time.time()-start_ep)),
                                        tt=int(np.round((episodes * np.mean(frames_stats)) / ((agent.step - step) / (time.time() - start)))),
                                        eps=np.round(agent.epsilon, 4),
                                        r=np.round(np.mean(rewards[-20:]), 4),    # [-100:]
                                        xp=int(np.round(np.mean(x_positions[-20:]))),
                                        xm=max_x_pos))
        start_ep = time.time()
        step = agent.step
    '''
    # SAVE WEIGHTS
    if (e % save_weights_every == 0 and e != 0):
        model_file_path = "./{}__{}__{}_online.HDF5".format(env_name, training_version_name, movement_type) # ,episode_count
        agent.model_online.save(model_file_path)
        model_target_file_path = "./{}__{}__{}_target.HDF5".format(env_name, training_version_name, movement_type)  # ,episode_count
        agent.model_target.save(model_target_file_path)
        print("weights saved")
    '''
    # Log rewards
    if(e % stats_every == 0 and e != 0):
        stats_ep_rewards['ep'].append(e)
        stats_ep_rewards['avg'].append(np.mean(rewards[-20:]))
        stats_ep_rewards['x_pos'].append(np.mean(x_positions[-20:]))

    # SAVE REWARDS
    if (e % save_rewards_every == 0 and e != 0):
        EPISODES_NAME = "{}__{}__{}__stats_ep".format(env_name, training_version_name, movement_type)
        REWARDS_NAME = "{}__{}__{}__stats_avg".format(env_name, training_version_name, movement_type)
        X_POS_NAME = "{}__{}__{}__stats_x_pos".format(env_name, training_version_name, movement_type)
        storage_agent.save_np(name=EPISODES_NAME, data=np.array(stats_ep_rewards['ep']), visualise=False)
        storage_agent.save_np(name=REWARDS_NAME, data=np.array(stats_ep_rewards['avg']), visualise=False)
        storage_agent.save_np(name=X_POS_NAME, data=np.array(stats_ep_rewards['x_pos']), visualise=False)
