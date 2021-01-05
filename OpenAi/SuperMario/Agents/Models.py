
import tensorflow as tf
import os
from tensorflow.keras import layers

import numpy as np


# from example:
# https://github.com/yingshaoxo/ML/tree/master/12.reinforcement_learning_with_mario_bros




class Simple_NN:
    def __init__(self, env, model_file_path, show_model = False):

        self.num_states = env.observation_space.shape
        self.num_actions = env.action_space.n

        self.show_model = show_model

        if os.path.exists(model_file_path):
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


"""
Here we define the Actor and Critic networks. These are basic Dense models
with `ReLU` activation.
Note: We need the initialization for last layer of the Actor to be between
`-0.003` and `0.003` as this prevents us from getting `1` or `-1` output values in
the initial stages, which would squash our gradients to zero,
as we use the `tanh` activation.
"""

class Models:
    def __init__(self, env, critic_lr = 0.002, actor_lr = 0.001):

        self.num_states = env.observation_space.shape[0]
        self.num_actions = env.action_space.n

        self.actor_model = self.get_actor()
        self.critic_model = self.get_critic()

        self.target_actor = self.get_actor()
        self.target_critic = self.get_critic()

        ##
        print("Actor Model: ")
        self.actor_model.summary()

        print("Critic Model: ")
        self.critic_model.summary()
        ##

        # Making the weights equal initially
        self.target_actor.set_weights(self.actor_model.get_weights())
        self.target_critic.set_weights(self.critic_model.get_weights())

        # Learning rate for actor-critic models
        self.critic_lr = critic_lr
        self.actor_lr = actor_lr

        self.critic_optimizer = tf.keras.optimizers.Adam(self.critic_lr)
        self.actor_optimizer = tf.keras.optimizers.Adam(self.actor_lr)

    def get_actor(self):

        inputs = layers.Input(shape=(self.num_states,))
        out = layers.Dense(256, activation="relu")(inputs)
        out = layers.Dense(256, activation="relu")(out)
        outputs = layers.Dense(self.num_actions)(out)


        model = tf.keras.Model(inputs, outputs)

        return model


    def get_critic(self):
        # State as input
        state_input = layers.Input(shape=(self.num_states))
        state_out = layers.Dense(16, activation="relu")(state_input)
        state_out = layers.Dense(32, activation="relu")(state_out)

        # Action as input
        action_input = layers.Input(shape=(self.num_actions))
        action_out = layers.Dense(32, activation="relu")(action_input)

        # Both are passed through seperate layer before concatenating
        concat = layers.Concatenate()([state_out, action_out])

        out = layers.Dense(256, activation="relu")(concat)
        out = layers.Dense(256, activation="relu")(out)
        outputs = layers.Dense(1)(out)

        # Outputs single value for give state-action
        model = tf.keras.Model([state_input, action_input], outputs)

        return model


    """
    `policy()` returns an action sampled from our Actor network plus some noise for
    exploration.
    """


    def policy(self, state, noise_object):
        sampled_actions = tf.squeeze(self.actor_model(state))
        noise = noise_object()
        # Adding noise to action
        sampled_actions = sampled_actions.numpy() + noise

        # We make sure action is within bounds
        #legal_action = np.clip(sampled_actions, self.lower_bound, self.upper_bound)

        #return [np.squeeze(legal_action)]

