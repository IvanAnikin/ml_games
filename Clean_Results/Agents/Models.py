
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from keras.optimizers import Adam


from typing import Tuple
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




class Actor_Critic_2():
    def __init__(self, input_shape, num_hidden, num_actions):

        inputs = layers.Input(shape=input_shape)
        flatten = layers.Flatten()(inputs)

        common = layers.Dense(num_hidden, activation="relu")(flatten)

        action = layers.Dense(num_actions, activation="softmax")(common)
        critic = layers.Dense(1)(common)

        self.model = keras.Model(inputs=inputs, outputs=[action, critic])



class ActorCritic_1(tf.keras.Model):
  """Combined actor-critic network."""

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

    def policy(self, state, noise_object):
        sampled_actions = tf.squeeze(self.actor_model(state))
        noise = noise_object()
        # Adding noise to action
        sampled_actions = sampled_actions.numpy() + noise

        # We make sure action is within bounds
        #legal_action = np.clip(sampled_actions, self.lower_bound, self.upper_bound)

        #return [np.squeeze(legal_action)]

