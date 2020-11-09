import time
import numpy as np

import gym


from tensorflow import keras

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout


def prep():

    #for _ in range(10):
    #    print(np.random.uniform(-2,2))

    env = gym.make("Pendulum-v0")

    env.reset()
    for step in range(100):
        env.render()

        observation, reward, done, info = env.step([np.random.uniform(-2,2)])

        if done:
            break


def create_model(LR = 1e-3, dropout = 0.4):
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



def test_games(num_trials = 1000, sim_steps = 200):

    env = gym.make("Pendulum-v0")

    max_reward = -10000
    best_game_memory = []

    for _ in range(num_trials):
        observation = env.reset()

        trial_reward = 0

        game_memory = []

        for step in range(sim_steps):
            #env.render()


            #action = np.random.randint(0, 1)
            action = np.random.uniform(-2,2)

            observation, reward, done, wtff = env.step([action])

            game_memory.append([observation, action])

            #print("observation: ", observation)
            #print("reward:", reward)
            #time.sleep(1)

            trial_reward += reward


        if (trial_reward > max_reward):
            max_reward = trial_reward
            best_game_memory = game_memory




        #print("trial_reward: ", trial_reward)

    print("max_reward: ", max_reward)
    print("best_game_memory: ", best_game_memory)


def training_data(num_trials = 1000, min_score = -1000, sim_steps = 200):
    env = gym.make("Pendulum-v0")

    trainingX, trainingY = [], []
    max_reward = -10000
    best_game_memory = []

    scores = []

    for _ in range(num_trials):
        observation = env.reset()

        trial_reward = 0
        game_memory = []
        training_sampleX, training_sampleY = [], []

        for step in range(sim_steps):

            action = np.random.uniform(-2, 2)                       # --> From NN

            observation, reward, done, wtff = env.step([action])
            game_memory.append([observation, action])
            trial_reward += reward

            training_sampleX.append(observation)
            training_sampleY.append(action)

            if(done):
                #print("done || step: ", step, "observation: ", observation, "reward: ", reward, "trial_reward: ", trial_reward)
                break

        if (trial_reward > max_reward and trial_reward > min_score):
            max_reward = trial_reward
            best_game_memory = game_memory

            trainingX += training_sampleX
            trainingY += training_sampleY

        scores.append(trial_reward)
                                                                    # Change NN

    #print(len(trainingY))
    #print("trainingX[0]): ", trainingX[0])
    #print("trainingY[0]): ", trainingY[0])

    trainingX, trainingY = np.array(trainingX), np.array(trainingY)
    print("Average: {}".format(np.mean(scores)))
    print("Median: {}".format(np.median(scores)))
    return trainingX, trainingY





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





def pendulum_v0():
    start_time = time.time()

    trainingX, trainingY = training_data(1000)  ##########################

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
    # model = keras.models.load_model("CartPoleModel_1")
    model_creation_time = time.time() - training_time_moment
    model_creation_time_moment = time.time()
    # print("Model creating time:", model_creation_time)
    # print()

    model.summary()
    print()