
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT, RIGHT_ONLY
from nes_py.wrappers import JoypadSpace

import tensorflow as tf

import os
import numpy as np
import matplotlib.pyplot as plt

#import OpenAi.SuperMario.Agents.Agent as Agent_file
#import Clean_Results.Agents.storage_agent as storage_agent
#import OpenAi.SuperMario.Agents.visualisation_agent as visualiser

# from example:
# https://github.com/yingshaoxo/ML/tree/master/12.reinforcement_learning_with_mario_bros

def save_np(name, data):

    np.save(name, data)

    load_name = name + ".npy"
    print("saved successfully, first array element: ", np.load(load_name)[1])

class Simple_NN:
    def __init__(self, env, model_file_path, load_model = False, show_model = False):

        self.num_states = env.observation_space.shape
        self.num_actions = env.action_space.n

        self.show_model = show_model

        if load_model and os.path.exists(model_file_path):
            print("loading model: {}".format(model_file_path))
            self.model  = tf.keras.models.load_model(model_file_path)
        else:
            self.model = self.generate_model()


        if self.show_model: self.model.summary()



    def generate_model(self):
        # + shape of hidden layers - try few
        model = tf.keras.models.Sequential([
            tf.keras.layers.Convolution2D(32, 8, 8, input_shape=self.num_states),
            tf.keras.layers.Activation('relu'),

            tf.keras.layers.Convolution2D(64, 4, 4),
            tf.keras.layers.Activation('relu'),

            tf.keras.layers.Convolution2D(64, 3, 3),
            tf.keras.layers.Activation('relu'),

            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(512),
            tf.keras.layers.Activation('relu'),

            tf.keras.layers.Dense(self.num_actions, activation=tf.nn.softmax),
            # tf.keras.layers.Dense(1, activation="linear")
        ])

        # + loss, metrics accuracy -- ?
        model.compile(optimizer='adam',
                      loss='mse',
                      metrics=['accuracy'])

        return model


class Agent_Simple_NN:
    def __init__(self, env, model_file_path, load_model = False, show_model = True):

        self.env = env

        self.num_states = self.env.observation_space.shape
        self.num_actions = self.env.action_space.n

        self.models = Simple_NN(env=env, model_file_path=model_file_path, load_model=load_model, show_model=show_model) # learning rate (other params)


    def get_action(self, prev_state, epsilon):

        if np.random.random() > epsilon:
            action = np.argmax(self.models.model.predict(np.expand_dims(prev_state, axis=0)))
        else:
            action = self.env.action_space.sample()

        return action


episodes = 30000
max_steps_per_episode = 1000
stats_every = 100
#show_every = 200
#show_length = 50
#show_until = 50
show_model = True
load_model = True

epsilon = 0.1


env_name = "SuperMarioBros-v0"
training_version_name = "Simple_NN"
env = gym_super_mario_bros.make(env_name)
#env = JoypadSpace(env, SIMPLE_MOVEMENT)
env = JoypadSpace(env, RIGHT_ONLY)

EPISODES_NAME = "env-{}__v-{}__ep-{}__stats-{}__episodes".format(env_name, training_version_name, episodes, stats_every)
REWARDS_NAME = "env-{}__v-{}__ep-{}__stats-{}__rewards".format(env_name, training_version_name, episodes, stats_every)
episodes = 10000
model_file_path = "./env-{}__v-{}__ep-{}__stats-{}__nn_model_2.HDF5".format(env_name, training_version_name, episodes, stats_every)

#visualiser.show_env_props(env)

print("episodes: ", episodes)
print("stats_every: ", stats_every)
print("epsilon: ", epsilon)
print("REWARDS_NAME: ", REWARDS_NAME)
print("load_model: ", load_model,  " || show_model: ", show_model)


Agent = Agent_Simple_NN(env=env, model_file_path=model_file_path, load_model=load_model, show_model=show_model)


ep_reward_list = []
avg_reward_list = []
ep_stats = []

done = True
last_state = None
identity = np.identity(env.action_space.n) # for quickly get a hot vector, like 0001000000000000


#for ep in range(episodes):

prev_state = env.reset()
episodic_reward = 0
avg_ep_reward = 0
avg_reward = 0
ep_trained = 0

episodes = 30000
for step in range(episodes):
    if done or step % max_steps_per_episode == 0 and step != 0:
        state = env.reset()
        print("done=True on step: ", step)

    action = Agent.get_action(prev_state=prev_state, epsilon=epsilon)
    state, reward, done, info = env.step(action)

    # save more rewards for batch training - if reward of batch better than other average rewards -> train on it
    if reward > avg_reward and reward > 0:
        Agent.models.model.train_on_batch(x=np.expand_dims(prev_state, axis=0), y=identity[action: action + 1])
        ep_trained += 1

    prev_state = state

    episodic_reward += reward
    if step % stats_every == 0 and step != 0:
        avg_ep_reward = episodic_reward/stats_every
        avg_reward = (avg_reward + avg_ep_reward) / 2
        ep_reward_list.append(avg_ep_reward)
        ep_stats.append(step)
        print("step: ", step, " || avg_ep_reward: ", avg_ep_reward, " || avg_reward: ", avg_reward, " || ep_trained: ", ep_trained)
        episodic_reward = 0
        ep_trained = 0
    #env.render()

#model_file_path = "./env-{}__v-{}__ep-{}__stats-{}__nn_model_2.HDF5".format(env_name, training_version_name, episodes, stats_every)
Agent.models.model.save(model_file_path)

# save ep_reward_list array
save_np(name=EPISODES_NAME, data=np.array(ep_stats))
save_np(name=REWARDS_NAME, data=np.array(ep_reward_list))



