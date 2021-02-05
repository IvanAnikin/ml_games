

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

import OpenAi.Altari_Games_GAI.Agents.Models as Models
import OpenAi.Altari_Games_GAI.Agents.hyperparameters as hp



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


        Models_class = Models.DQN(self.states, num_hidden, self.num_actions, show_model,
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
            target_q = reward + (1. - done) * self.gamma * next_q[np.arange(0, self.batch_size), a]
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
