import time
import numpy as np
import random

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

        print("observation: ", observation)
        print("reward:", reward)
        time.sleep(1)

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



def test_games(num_trials = 1000, sim_steps = 199):

    env = gym.make("Pendulum-v0")

    max_reward = -10000
    least_velocity = 10000

    velocities = []

    for _ in range(num_trials):
        observation = env.reset()

        trial_reward = 0
        trial_velocity = 0


        for step in range(sim_steps):


            action = np.random.uniform(-2,2)

            observation, reward, done, wtff = env.step([action])


            trial_velocity += abs(observation[2])
            trial_reward += reward

            if(done):
                break

        velocities.append(trial_velocity)

        if (trial_reward > max_reward):
            max_reward = trial_reward

        if (trial_velocity < least_velocity):
            least_velocity = trial_velocity



    #print("max_reward: ", max_reward)
    print("Average velocity: {}".format(np.mean(velocities)))
    print("Median velocity: {}".format(np.median(velocities)))


def training_data(num_trials = 1000, min_score = -900, min_velocity = 700, sim_steps = 199):
    env = gym.make("Pendulum-v0")

    print("num_trials: ", num_trials, "|| min_score: ", min_score, "|| min_velocity: ", min_velocity)

    trainingX, trainingY = [], []
    max_reward = -10000

    scores = []
    velocities = []
    SHOW_EVERY = 200

    for episode in range(num_trials):
        observation = env.reset()

        trial_reward = 0
        trial_velocity = 0
        game_memory = []
        training_sampleX, training_sampleY = [], []

        for step in range(sim_steps):

            action = np.random.uniform(-2, 2)                       # --> From NN

            observation, reward, done, wtff = env.step([action])
            game_memory.append([observation, action])
            trial_reward += reward
            trial_velocity += abs(observation[2])

            training_sampleX.append(observation)
            training_sampleY.append(action)

            if(done):
                print("WON")
                break

        if (trial_reward > max_reward): max_reward = trial_reward

        if (trial_reward > min_score and trial_velocity < min_velocity):

            trainingX += training_sampleX
            trainingY += training_sampleY

            scores.append(trial_reward)
            velocities.append(trial_velocity)


        if episode % SHOW_EVERY == 0: print("episode: ", episode, "saved steps: ", len(trainingY))

    #print(len(trainingY))
    #print("trainingX[0]): ", trainingX[0])
    #print("trainingY[0]): ", trainingY[0])

    trainingX, trainingY = np.array(trainingX), np.array(trainingY)
    print("Saved games: ", trainingX.shape)
    print("Best score: ", max_reward)
    print("Average score: {}".format(np.mean(scores)))
    print("Median score: {}".format(np.median(scores)))
    print("Average velocity: {}".format(np.mean(velocities)))
    print("Median velocity: {}".format(np.median(velocities)))


    return trainingX, trainingY





def play_and_train(num_trials = 2000, min_score = -900, min_velocity = 700, sim_steps = 199):
    env = gym.make("Pendulum-v0")

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

    model_creation_time = time.time() - training_time_moment
    model_creation_time_moment = time.time()

    model.summary()
    print()

    epochs = 10
    history = model.fit(trainingX, trainingY, epochs=epochs)  ##########################
    model_fit_time = time.time() - model_creation_time_moment
    print()
    print("Model fitting time:", model_fit_time)
    print()

    print("num_trials: ", num_trials, "|| min_score: ", min_score, "|| min_velocity: ", min_velocity)

    trainingX, trainingY = [], []
    max_reward = -10000
    max_velocity = 10000

    scores = []
    velocities = []

    SHOW_EVERY = 10




    for episode in range(num_trials):
        observation = env.reset()

        trial_reward = 0
        trial_velocity = 0
        game_memory = []
        training_sampleX, training_sampleY = [], []

        for step in range(sim_steps):

            if episode % SHOW_EVERY == 0 and episode != 0: env.render()

            if len(observation) == 0:
                action = random.randrange(0, 2)
            else:
                action = model.predict(observation.reshape(1, 3))                              # --> From NN

            observation, reward, done, wtff = env.step([action])
            game_memory.append([observation, action])
            trial_reward += reward
            trial_velocity += abs(observation[2])

            training_sampleX.append(observation)
            training_sampleY.append(action)

            if (done):
                print("WON")
                break

        if (trial_reward > max_reward): max_reward = trial_reward

        if (trial_reward > min_score and trial_velocity < min_velocity and trial_reward > max_reward and trial_velocity < max_velocity):
            trainingX += training_sampleX
            trainingY += training_sampleY

            scores.append(trial_reward)
            velocities.append(trial_velocity)

        if episode % SHOW_EVERY == 0 and episode != 0: print("episode: ", episode, "saved steps: ", len(trainingY))

            # Change NN

    # print(len(trainingY))
    # print("trainingX[0]): ", trainingX[0])
    # print("trainingY[0]): ", trainingY[0])

    trainingX, trainingY = np.array(trainingX), np.array(trainingY)
    print("Saved steps: ", trainingX.shape)
    print("Best score: ", max_reward)
    print("Average score: {}".format(np.mean(scores)))
    print("Median score: {}".format(np.median(scores)))
    print("Average velocity: {}".format(np.mean(velocities)))
    print("Median velocity: {}".format(np.median(velocities)))

    return trainingX, trainingY








def create_model(LR, dropout):
    model = Sequential()
    model.add(Dense(128, input_shape=(3,), activation="relu"))
    model.add(Dropout(dropout))

    model.add(Dense(256, activation="relu"))
    model.add(Dropout(dropout))

    model.add(Dense(512, activation="relu"))
    model.add(Dropout(dropout))

    model.add(Dense(256, activation="relu"))
    model.add(Dropout(dropout))

    model.add(Dense(128, activation="relu"))
    model.add(Dropout(dropout))
    model.add(Dense(1, activation="softmax"))

    model.compile(
        loss="categorical_crossentropy",
        optimizer=keras.optimizers.Adam(lr=LR),                                           #define LR !!!!!!!!!
        metrics=["accuracy"])
    return model





def pendulum_v0():
    start_time = time.time()

    trainingX, trainingY = training_data(10000, -900, 700)  ##########################

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

    epochs = 10
    history = model.fit(trainingX, trainingY, epochs=epochs)  ##########################
    model_fit_time = time.time() - model_creation_time_moment
    print()
    print("Model fitting time:", model_fit_time)
    print()




    env = gym.make("Pendulum-v0")
    env.reset()

    game_memory = []
    observation = []
    score = 0
    for step in range(199):
        env.render()
        if len(observation) == 0:
            action = random.randrange(0, 2)
        else:
            action = model.predict(observation.reshape(1, 3))
        observation, reward, done, info = env.step([action])
        prev_obs = observation
        game_memory.append([observation, action])
        score += reward
        if done:
            break

    print("Score:", score)