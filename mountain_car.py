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
            action = np.random.choice([0,1,2])
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



def test_play_display_info(episodes_count = 1000):
    env = gym.make('MountainCar-v0')

    print("env.action_space.n: ", env.action_space.n)
    print("env.observation_space.shape[0]]: ", env.observation_space.shape[0])

    num_episode_steps = env.spec.max_episode_steps
    print("num_episode_steps: ", num_episode_steps)

    all_rewards = 0
    max_reward = 0

    for episode in range(episodes_count):

        total_reward = 0

        observation = env.reset()

        state = np.reshape(observation, [1, env.observation_space.shape[0]])

        if (episode % 100 == 0 and episode != 0): print("episode", episode)

        for episode_step in range(num_episode_steps):
            # env.render(mode="human")

            action = random.randrange(env.action_space.n)

            observation, reward, done, _ = env.step(action)

            # if(episode_step%40==0 and episode_step!=0):
            # print(episode_step, "||   reward: ", reward)
            # print("observation: ", observation)
            # print("state: ", state)

            # Recalculates the reward
            if observation[1] > state[0][1] >= 0 and observation[1] >= 0:
                total_reward += 10
            if observation[1] < state[0][1] <= 0 and observation[1] <= 0:
                total_reward += 10
            # if done and episode_step < num_episode_steps - 1:
            # print(3)
            else:
                total_reward-=10

            if(done):
                all_rewards += total_reward
                if(total_reward>max_reward):
                    max_reward = total_reward
                    #SAVE GAME
                #if (episode % 100 == 0 and episode != 0):
                #    print("total_reward", total_reward)


    average_reward = all_rewards/episodes_count
    print("average_reward: ", average_reward)
    print("max_reward: ", max_reward)

