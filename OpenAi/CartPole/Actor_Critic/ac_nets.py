import os

import numpy as np
from collections import deque
from typing import Any, List, Sequence, Tuple


import keras as keras
import tensorflow as tf

from keras.models import Sequential, Model
from keras import layers
from keras.layers import Dense, Dropout, Input
from keras.layers.merge import Add, Multiply
from keras.optimizers import Adam



#https://github.com/llSourcell/actor_critic/blob/master/actor_critic.py
class ActorCritic:
    def __init__(self, env, alpha = 0.0001, gamma = 0.95, tau = 0.125, epsilon = 1., epsilon_decay = 0.995):
        self.env = env

        self.alpha = alpha
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.gamma = gamma
        self.tau = tau

        self.memory = deque(maxlen=2000)


        self.state_input, self.actor_model = self.create_actor_model()

        self.critic_state_input, self.critic_action_input, self.critic_model = self.create_critic_model()
        self.critic_model.summary()


    #actor nn creation
    def create_actor_model(self):
        state_input = Input(shape=self.env.observation_space.shape)

        model = Sequential()
        model.add(Dense(24, activation="relu"))
        model.add(Dense(48, activation="relu"))
        model.add(Dense(24, activation="relu"))
        model.add(Dense(self.env.action_space.shape[0], activation="relu"))

        model.compile(
            loss="mse",
            optimizer=Adam(lr=self.alpha))

        return state_input, model

    #critic model creation
    def create_critic_model(self):
        state_input = Input(shape=self.env.observation_space.shape)
        state_h1 = Dense(24, activation='relu')(state_input)
        state_h2 = Dense(48)(state_h1)

        action_input = Input(shape=self.env.action_space.shape)
        action_h1 = Dense(48)(action_input)

        merged = Add()([state_h2, action_h1])
        merged_h1 = Dense(24, activation='relu')(merged)
        output = Dense(1, activation='relu')(merged_h1)
        model = Model([state_input, action_input], output)

        adam = Adam(lr=0.001)
        model.compile(loss="mse", optimizer=adam)
        return state_input, action_input, model



    def train(self, samples, batch_size):

        if len(self.memory) < batch_size:
            return

        #train
        self.train_actor(samples)
        self.train_critic(samples)

        return

    def train_actor(self, samples):

        for sample in samples:
            observation, action, reward, observation_, done = sample

            #calculate grads

            #optimize actor model

        return

    def train_critic(self, samples):

        for sample in samples:
            observation, action, reward, observation_, done = sample


        return


    def act(self, observation):
        self.epsilon *= self.epsilon_decay
        if np.random.random() < self.epsilon:
            return self.env.action_space.sample()
        return self.actor_model.predict(observation)


#https://www.tensorflow.org/tutorials/reinforcement_learning/actor_critic
class ActorCritic_v2:
    def __init__(
            self,
            num_actions: int,
            num_hidden_units: int):
        """Initialize."""
        super().__init__()

        self.common = layers.Dense(num_hidden_units, activation="relu")
        self.actor = layers.Dense(num_actions)
        self.critic = layers.Dense(1)

    def call(self, inputs: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        x = self.common(inputs)
        return self.actor(x), self.critic(x)


class ActorCritic_v3(keras.Model):
    def __init__(self, n_actions, fc1_dims=1024, fc2_dims=512,
            name='actor_critic', chkpt_dir='tmp/actor_critic'):
        super(ActorCritic_v3, self).__init__()
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        self.model_name = name
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name+'_ac')

        self.fc1 = Dense(self.fc1_dims, activation='relu')
        self.fc2 = Dense(self.fc2_dims, activation='relu')
        self.pi = Dense(n_actions, activation='softmax')
        self.v = Dense(1, activation=None)

    def call(self, state):
        value = self.fc1(state)
        value = self.fc2(value)

        v = self.v(value)
        pi = self.pi(value)

        return v, pi