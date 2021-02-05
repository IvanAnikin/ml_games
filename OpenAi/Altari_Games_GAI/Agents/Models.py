
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

