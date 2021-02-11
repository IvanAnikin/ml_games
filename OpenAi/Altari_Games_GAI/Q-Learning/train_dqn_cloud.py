

import time
import numpy as np
from PIL import Image

import gym
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT, RIGHT_ONLY
from nes_py.wrappers import JoypadSpace


# WRAPPER

import numpy as np
import os

os.environ.setdefault('PATH', '')
from collections import deque
import gym
from gym import spaces
import cv2

cv2.ocl.setUseOpenCL(False)


class NoopResetEnv(gym.Wrapper):
    def __init__(self, env, noop_max=30):
        """Sample initial states by taking random number of no-ops on reset.
        No-op is assumed to be action 0.
        """
        gym.Wrapper.__init__(self, env)
        self.noop_max = noop_max
        self.override_num_noops = None
        self.noop_action = 0
        assert env.unwrapped.get_action_meanings()[0] == 'NOOP'

    def reset(self, **kwargs):
        """ Do no-op action for a number of steps in [1, noop_max]."""
        self.env.reset(**kwargs)
        if self.override_num_noops is not None:
            noops = self.override_num_noops
        else:
            noops = self.unwrapped.np_random.randint(1, self.noop_max + 1)  # pylint: disable=E1101
        assert noops > 0
        obs = None
        for _ in range(noops):
            obs, _, done, _ = self.env.step(self.noop_action)
            if done:
                obs = self.env.reset(**kwargs)
        return obs

    def step(self, ac):
        return self.env.step(ac)


class FireResetEnv(gym.Wrapper):
    def __init__(self, env):
        """Take action on reset for environments that are fixed until firing."""
        gym.Wrapper.__init__(self, env)
        assert env.unwrapped.get_action_meanings()[1] == 'FIRE'
        assert len(env.unwrapped.get_action_meanings()) >= 3

    def reset(self, **kwargs):
        self.env.reset(**kwargs)
        obs, _, done, _ = self.env.step(1)
        if done:
            self.env.reset(**kwargs)
        obs, _, done, _ = self.env.step(2)
        if done:
            self.env.reset(**kwargs)
        return obs

    def step(self, ac):
        return self.env.step(ac)


class EpisodicLifeEnv(gym.Wrapper):
    def __init__(self, env):
        """Make end-of-life == end-of-episode, but only reset on true game over.
        Done by DeepMind for the DQN and co. since it helps value estimation.
        """
        gym.Wrapper.__init__(self, env)
        self.lives = 0
        self.was_real_done = True

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self.was_real_done = done
        # check current lives, make loss of life terminal,
        # then update lives to handle bonus lives
        lives = self.env.unwrapped.ale.lives()
        if lives < self.lives and lives > 0:
            # for Qbert sometimes we stay in lives == 0 condition for a few frames
            # so it's important to keep lives > 0, so that we only reset once
            # the environment advertises done.
            done = True
        self.lives = lives
        return obs, reward, done, info

    def reset(self, **kwargs):
        """Reset only when lives are exhausted.
        This way all states are still reachable even though lives are episodic,
        and the learner need not know about any of this behind-the-scenes.
        """
        if self.was_real_done:
            obs = self.env.reset(**kwargs)
        else:
            # no-op step to advance from terminal/lost life state
            obs, _, _, _ = self.env.step(0)
        self.lives = self.env.unwrapped.ale.lives()
        return obs


class MaxAndSkipEnv(gym.Wrapper):
    def __init__(self, env, skip=4):
        """Return only every `skip`-th frame"""
        gym.Wrapper.__init__(self, env)
        # most recent raw observations (for max pooling across time steps)
        self._obs_buffer = np.zeros((2,) + env.observation_space.shape, dtype=np.uint8)
        self._skip = skip

    def step(self, action):
        """Repeat action, sum reward, and max over last observations."""
        total_reward = 0.0
        done = None
        for i in range(self._skip):
            obs, reward, done, info = self.env.step(action)
            if i == self._skip - 2: self._obs_buffer[0] = obs
            if i == self._skip - 1: self._obs_buffer[1] = obs
            total_reward += reward
            if done:
                break
        # Note that the observation on the done=True frame
        # doesn't matter
        max_frame = self._obs_buffer.max(axis=0)

        return max_frame, total_reward, done, info

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)


class ClipRewardEnv(gym.RewardWrapper):
    def __init__(self, env):
        gym.RewardWrapper.__init__(self, env)

    def reward(self, reward):
        """Bin reward to {+1, 0, -1} by its sign."""
        return np.sign(reward)


class WarpFrame(gym.ObservationWrapper):
    def __init__(self, env, width=84, height=84, grayscale=True):
        """Warp frames to 84x84 as done in the Nature paper and later work."""
        gym.ObservationWrapper.__init__(self, env)
        self.width = width
        self.height = height
        self.grayscale = grayscale
        if self.grayscale:
            self.observation_space = spaces.Box(low=0, high=255,
                                                shape=(self.height, self.width, 1), dtype=np.uint8)
        else:
            self.observation_space = spaces.Box(low=0, high=255,
                                                shape=(self.height, self.width, 3), dtype=np.uint8)

    def observation(self, frame):
        if self.grayscale:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        frame = cv2.resize(frame, (self.width, self.height), interpolation=cv2.INTER_AREA)
        if self.grayscale:
            frame = np.expand_dims(frame, -1)
        return frame


class LazyFrames(object):
    def __init__(self, frames):
        """This object ensures that common frames between the observations are only stored once.
        It exists purely to optimize memory usage which can be huge for DQN's 1M frames replay
        buffers.
        This object should only be converted to numpy array before being passed to the model.
        You'd not believe how complex the previous solution was."""
        self._frames = frames
        self._out = None

    def _force(self):
        if self._out is None:
            self._out = np.concatenate(self._frames, axis=-1)
            self._frames = None
        return self._out

    def __array__(self, dtype=None):
        out = self._force()
        if dtype is not None:
            out = out.astype(dtype)
        return out

    def __len__(self):
        return len(self._force())

    def __getitem__(self, i):
        return self._force()[i]


class FrameStack(gym.Wrapper):
    def __init__(self, env, k):
        """Stack k last frames.
        Returns lazy array, which is much more memory efficient.
        See Also
        --------
        baselines.common.atari_wrappers.LazyFrames
        """
        gym.Wrapper.__init__(self, env)
        self.k = k
        self.frames = deque([], maxlen=k)
        shp = env.observation_space.shape
        self.observation_space = spaces.Box(low=0, high=255, shape=(shp[:-1] + (shp[-1] * k,)),
                                            dtype=env.observation_space.dtype)

    def reset(self):
        ob = self.env.reset()
        for _ in range(self.k):
            self.frames.append(ob)
        return self._get_ob()

    def step(self, action):
        ob, reward, done, info = self.env.step(action)
        self.frames.append(ob)
        return self._get_ob(), reward, done, info

    def _get_ob(self):
        assert len(self.frames) == self.k
        return LazyFrames(list(self.frames))


def wrapper(env, states):
    """Apply a common set of wrappers for Atari games."""
    # env = EpisodicLifeEnv(env)
    # env = NoopResetEnv(env, noop_max=10)

    width = states[0]
    height = states[1]
    skip = states[2]

    env = MaxAndSkipEnv(env, skip=skip)
    if 'FIRE' in env.unwrapped.get_action_meanings():
        env = FireResetEnv(env)
    env = WarpFrame(env, width=width, height=height)
    env = FrameStack(env, 4)
    env = ClipRewardEnv(env)
    return env

##################################

# AGENT


import tensorflow as tf
from tensorflow import keras
from keras import layers
from keras.optimizers import Adam
from keras.losses import Huber

from typing import Tuple, List
import numpy as np
from collections import deque
import random
import os


class DQN():
    def __init__(self, input_shape, num_hidden, num_actions, show_model, load_model, model_file_path_online, model_file_path_target):

        self.states = input_shape
        self.num_hidden = num_hidden
        self.num_actions = num_actions

        if (load_model and os.path.exists(model_file_path_online) and os.path.exists(model_file_path_target)):
            print("loading model: {}".format(model_file_path_online))
            self.model_online = tf.keras.models.load_model(model_file_path_online)
            print("loading model: {}".format(model_file_path_target))
            self.model_target = tf.keras.models.load_model(model_file_path_target)
        else:
            self.model_online = self.generate_model()
            self.model_target = self.generate_model()

            # MODELS VISUALISATION
        if (show_model):
            print("Model 'online' summary: ")
            self.model_online.summary()
            print("Model 'target': same structure")


    def generate_model(self):

        inputs = layers.Input(shape=self.states)
        input_float = tf.cast(inputs, tf.float32) / 255.

        conv_1 = layers.Convolution2D(filters=32, kernel_size=8, strides=4, activation="relu")(input_float)
        conv_2 = layers.Convolution2D(filters=32, kernel_size=4, strides=2, activation="relu")(conv_1)
        conv_3 = layers.Convolution2D(filters=32, kernel_size=3, strides=1, activation="relu")(conv_2)

        flatten = layers.Flatten()(conv_3)

        common = layers.Dense(self.num_hidden, activation="relu")(flatten)

        actions = layers.Dense(self.num_actions, activation="linear")(common)

        model = keras.Model(inputs=inputs, outputs=actions)
        model.compile(loss="mse", optimizer=Adam(lr=0.001), metrics=['accuracy'])

        return model

    def generate_model_2(self):

        self.inputs = layers.Input(shape=self.states)
        self.input_float = tf.cast(self.inputs, tf.float32) / 255.

        # ONLINE
        self.conv_1 = layers.Convolution2D(filters=32, kernel_size=8, strides=4, activation="relu")(self.input_float)
        self.conv_2 = layers.Convolution2D(filters=32, kernel_size=4, strides=2, activation="relu")(self.conv_1)
        self.conv_3 = layers.Convolution2D(filters=32, kernel_size=3, strides=1, activation="relu")(self.conv_2)
        self.flatten = layers.Flatten()(self.conv_3)
        self.common = layers.Dense(self.num_hidden, activation="relu")(self.flatten)
        self.actions = layers.Dense(self.num_actions, activation="linear")(self.common)

        # TARGET
        self.conv_1_t = layers.Convolution2D(filters=32, kernel_size=8, strides=4, activation="relu")(self.input_float)
        self.conv_2_t = layers.Convolution2D(filters=32, kernel_size=4, strides=2, activation="relu")(self.conv_1_t)
        self.conv_3_t = layers.Convolution2D(filters=32, kernel_size=3, strides=1, activation="relu")(self.conv_2_t)
        self.flatten_t = layers.Flatten()(self.conv_3_t)
        self.common_t = layers.Dense(self.num_hidden, activation="relu")(self.flatten_t)
        self.actions_t = layers.Dense(self.num_actions, activation="linear")(self.common_t)


class DQN_Agent():
    def __init__(self, env, states, num_hidden, epsilon, eps_decay, eps_min,
                 max_memory, copy, learn_each, save_each, batch_size, gamma,
                 show_model, load_model, model_file_path_online, model_file_path_target, double_q):
        self.env = env
        self.states = states
        self.num_hidden = num_hidden
        self.epsilon = epsilon
        self.eps_decay = eps_decay
        self.eps_min = eps_min
        self.memory = deque(maxlen=max_memory)
        self.copy = copy
        self.batch_size = batch_size
        self.gamma = gamma

        self.step = 0
        self.learn_step = 0
        self.learn_each = learn_each
        self.save_each = save_each
        self.num_actions = env.action_space.n
        self.double_q = double_q                       # DQ - True


        Models_class = DQN(self.states, num_hidden, self.num_actions, show_model,
                                  load_model, model_file_path_online, model_file_path_target)
        self.model_online = Models_class.model_online
        self.model_target = Models_class.model_target



    def run(self, state):

        state = tf.convert_to_tensor(state)
        state = tf.expand_dims(state, 0)

        if np.random.rand() < self.epsilon:
            # Random action
            action = self.env.action_space.sample()
        else:
            # Policy action
            q = self.model_online(state)
            action = np.argmax(q)

        # Decrease eps
        self.epsilon *= self.eps_decay
        self.epsilon = max(self.eps_min, self.epsilon)
        # Increment step
        self.step += 1

        return action


    def learn(self):

        # Sync target network
        if self.step % self.copy == 0:
            self.copy_model()
        # Checkpoint model
        if self.step % self.save_each == 0:
            self.save_model()
                                                        # Break if burn-in
                                                        #if self.step < self.burnin:
                                                        #    return
        # Break if no training
        if self.learn_step < self.learn_each:
            self.learn_step += 1
            return
        # Sample batch
        if(len(self.memory) < self.batch_size): batch = random.sample(self.memory, len(self.memory))
        else: batch = random.sample(self.memory, self.batch_size)
        state, next_state, action, reward, done = map(np.array, zip(*batch))

        # Get next q values from target network
        next_q = self.model_target(next_state)
        # Calculate discounted future reward
        if self.double_q:
            q = self.model_online(next_state)
            a = np.argmax(q, axis=1)
            # target_q = reward + (1. - done) * self.gamma * next_q[np.arange(0, self.batch_size), a]
            if (len(self.memory) < self.batch_size):
                target_q = [0] * len(self.memory)
            else:
                target_q = [0] * self.batch_size
            index = 0
            for TQ in target_q:
                one_target_q = reward[index] + (1. - done[index]) * self.gamma * next_q[index, a[index]]
                target_q[index] = one_target_q
                index += 1
        else:
            target_q = reward + (1. - done) * self.gamma * np.amax(next_q, axis=1)


        self.a_true = np.array(action)
        self.q_true = np.array(target_q)
        current_states_q_values = self.model_online(state)

        X = state
        Y = np.array(current_states_q_values)

        index = 0
        for one_q_true in self.q_true:

            Y[index, self.a_true[index]] = one_q_true

            index+=1

        self.model_online.fit(X, Y, verbose=0) # verbose=0 -- logging none

        # Reset learn step
        self.learn_step = 0

        return

    def copy_model(self):

        self.model_target.set_weights(self.model_online.get_weights())

    def save_model(self):

        return

    def add(self, experience):

        self.memory.append(experience)

        return



#################################################


# STORAGE AGENT

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation


def save_np(name, data, visualise):

    np.save(name, data)

    load_name = name + ".npy"
    if(visualise): print("saved successfully, first array element: ", np.load(load_name)[1])


def load_np(name):
    load_name = name + ".npy"

    return np.load(load_name)

def save_frames_as_gif(frames, path='./', name="gym_animation"):
    filename = name + '.gif'

    # Mess with this to change frame size
    plt.figure(figsize=(frames[0].shape[1] / 72.0, frames[0].shape[0] / 72.0), dpi=72)

    patch = plt.imshow(frames[0])
    plt.axis('off')

    def animate(i):
        patch.set_data(frames[i])

    anim = animation.FuncAnimation(plt.gcf(), animate, frames=len(frames), interval=50)
    anim.save(path + filename, writer='imagemagick', fps=60)
    print("saved gif: {}".format(filename))

##################################################


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

episodes = 10000            # 10000
show_every = 5              # episodes
stats_every = 2             # episodes
save_weights_every = 10     # episodes
save_rewards_every = 10     # episodes

epsilon = 1                 # 1
eps_decay = 0.9999995       # 999999 - epsilon = 0.45 after 5000 episodes || 0.99999 - avg 0.4 epsilon left after 300 episodes (0.99999975)
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
        EPSILON_NAME = "{}__{}__{}__stats_epsilon".format(env_name, training_version_name, movement_type)
        save_np(name=EPISODES_NAME, data=np.array(stats_ep_rewards['ep']), visualise=False)
        save_np(name=REWARDS_NAME, data=np.array(stats_ep_rewards['avg']), visualise=False)
        save_np(name=STEPS_NAME, data=np.array(stats_ep_rewards['steps']), visualise=False)
        save_np(name=EPSILON_NAME, data=np.array(stats_ep_rewards['epsilon']), visualise=False)

