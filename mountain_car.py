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






def gather_data(env, num_trials, min_score, sim_steps):
    trainingX, trainingY = [], []

    print("num_trials: ", num_trials)
    print("min_score: ", min_score)
    print("sim_steps: ", sim_steps)
    print()
    #print(f"{bcolors.WARNING}Warning: No active frommets remain. Continue?{bcolors.ENDC}")  COLORED

    scores = []
    for _ in range(num_trials):
        observation = env.reset()
        score = 0
        training_sampleX, training_sampleY = [], []
        for step in range(sim_steps):
            # action corresponds to the prevgit remote add origin https://github.com/IvanAnikin/ml_games.gitious observation so record before step
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

    trainingX, trainingY = np.array(trainingX), np.array(trainingY)
    print("Average: {}".format(np.mean(scores)))
    print("Median: {}".format(np.median(scores)))
    return trainingX, trainingY



#def create_model(LR, dropout):


def training_data(num_trials, min_score, sim_steps):
    env = gym.make("MountainCar-v0")
    trainingX, trainingY = gather_data(env, num_trials, min_score, sim_steps)
    return trainingX, trainingY

