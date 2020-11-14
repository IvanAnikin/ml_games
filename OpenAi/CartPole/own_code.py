from tensorflow import keras


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

from tensorflow.keras.callbacks import TensorBoard



import time
import numpy as np
import matplotlib.pyplot as plt
import gym
import random


def gather_data(num_trials = 5000, min_score = 70, sim_steps = 500):
    env = gym.make("CartPole-v0")

    trainingX, trainingY = [], []
    SHOW_EVERY = 500

    print("num_trials: ", num_trials)
    print("min_score: ", min_score)
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

        if episode % SHOW_EVERY == 0: print("episode: ", episode, " ||  saved trials: ", len(trainingY))

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


def train_model(trainingX, trainingY):

    model = create_model(1e-3, 0.4)

    epochs = 10
    history = model.fit(trainingX, trainingY, epochs=epochs)




    return model, history, epochs

def visualise_fitting(history, epochs):
    loss_train = np.array(history.history['loss'])
    accuracy_train = np.array(history.history['accuracy'])

    epochs_range = range(1, epochs + 1)
    plt.plot(epochs_range, loss_train, 'g', label='Training loss')
    plt.title('Training loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    plt.plot(epochs_range, accuracy_train, 'g', label='Training accuracy')
    plt.title('Training accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()




def  play_trained_game(model, game_sim_steps = 500, game_num_trials=10):
    env = gym.make("CartPole-v0")

    score = 0

    observation = env.reset()
    for step in range(game_sim_steps):
        env.render()
        if len(observation) == 0:
            action = random.randrange(0, 2)
        else:
            action = np.argmax(model.predict(observation.reshape(1, 4)))
        observation, reward, done, info = env.step(action)
        score += reward
        if done:
            break

    return score


def own_cart_pole_main(num_trials = 5000, min_score = 80, sim_steps = 500):

    trainingX, trainingY = gather_data(num_trials, min_score, sim_steps)

    model, history, epochs = train_model(trainingX, trainingY)

    visualise_fitting(history, epochs)

    score = play_trained_game(model, 500)
    print("score: ", score)


