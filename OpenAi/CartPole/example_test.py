import gym
import numpy as np
import tensorflow as tf

from statistics import median, mean
from collections import Counter
from tensorflow import keras



import random
#import tflearn
#from tflearn.layers.core import input_data, dropout, fully_connected
#from tflearn.layers.estimator import regression
from statistics import median, mean
from collections import Counter


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

from tensorflow.keras.callbacks import TensorBoard



import time
import numpy as np
import matplotlib.pyplot as plt
import gym
import random






def gather_data(env, num_trials, min_score, sim_steps):
    trainingX, trainingY = [], []
    SHOW_EVERY = 500

    print("num_trials: ", num_trials)
    print("min_score: ", min_score)
    print("sim_steps: ", sim_steps)
    print()

    scores = []
    for episode in range(num_trials):
        observation = env.reset()
        score = 0
        training_sampleX, training_sampleY = [], []
        for step in range(sim_steps):
            # action corresponds to the previous observation so record before step
            action = np.random.randint(0, 2)
            one_hot_action = np.zeros(2)
            one_hot_action[action] = 1
            training_sampleX.append(observation)
            training_sampleY.append(one_hot_action)

            observation, reward, done, _ = env.step(action)
            score += reward
            if done:
                break
        if score > min_score:
            scores.append(score)
            trainingX += training_sampleX
            trainingY += training_sampleY

        if episode % SHOW_EVERY == 0: print("episode: ", episode, "saved trials: ", len(trainingY))

    trainingX, trainingY = np.array(trainingX), np.array(trainingY)
    print("Average: {}".format(np.mean(scores)))
    print("Median: {}".format(np.median(scores)))
    return trainingX, trainingY



#LR = 1e-3
#dropout = 0.4

def create_model(LR, dropout):
    model = Sequential()
    model.add(Dense(128, input_shape=(4,), activation="relu"))
    model.add(Dropout(dropout))

    model.add(Dense(256, activation="relu"))
    model.add(Dropout(dropout))

    model.add(Dense(512, activation="relu"))
    model.add(Dropout(dropout))

    model.add(Dense(256, activation="relu"))
    model.add(Dropout(dropout))

    model.add(Dense(128, activation="relu"))
    model.add(Dropout(dropout))
    model.add(Dense(2, activation="softmax"))

    model.compile(
        loss="categorical_crossentropy",
        optimizer=keras.optimizers.Adam(lr=LR),                                           #define LR !!!!!!!!!
        metrics=["accuracy"])
    return model




def training_data(num_trials, min_score, sim_steps):
    env = gym.make("CartPole-v0")
    trainingX, trainingY = gather_data(env, num_trials, min_score, sim_steps)
    return trainingX, trainingY







def cart_pole_v1():

    start_time = time.time()

    trainingX, trainingY = training_data(20000, 70, 500)  ##########################

    training_time = time.time() - start_time
    training_time_moment = time.time()
    print()
    print("training X: ", trainingX.shape)
    print("training Y: ", trainingY.shape)
    print()
    print()
    print("Getting training data time:", training_time)
    print()
    print()

    model = create_model(1e-3, 0.4)  ##########################
    #model = keras.models.load_model("CartPoleModel_1")
    model_creation_time = time.time() - training_time_moment
    model_creation_time_moment = time.time()
    # print("Model creating time:", model_creation_time)
    #print()

    model.summary()
    print()


    #Model saving and loading

    #print("saving model")
    #model.save("CartPoleModel_1")


    reconstructed_model = keras.models.load_model("CartPoleModel_1")
    print("Reconstructed model:")
    reconstructed_model.summary()
    print()

    epochs = 10
    history = model.fit(trainingX, trainingY, epochs=epochs)  ##########################
    model_fit_time = time.time() - model_creation_time_moment
    print()
    print("Model fitting time:", model_fit_time)
    print()



    loss_train = np.array(history.history['loss'])
    epochs_range = range(1, epochs + 1)
    plt.plot(epochs_range, loss_train, 'g', label='Training loss')
    plt.title('Training and Validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    env = gym.make("CartPole-v0")

    sim_steps = 500

    env.reset()
    game_memory = []
    observation = []
    score = 0
    for step in range(sim_steps):
        env.render()
        if len(observation) == 0:
            action = random.randrange(0, 2)
        else:
            action = np.argmax(model.predict(observation.reshape(1, 4)))
        observation, reward, done, info = env.step(action)
        prev_obs = observation
        game_memory.append([observation, action])
        score += reward
        if done:
            break

    print("Score:", score)