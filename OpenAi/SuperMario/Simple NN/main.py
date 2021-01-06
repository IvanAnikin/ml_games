
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT, RIGHT_ONLY
from nes_py.wrappers import JoypadSpace

import tensorflow as tf

import os
import numpy as np
import matplotlib.pyplot as plt

import OpenAi.SuperMario.Agents.Agent as Agent_file
import Clean_Results.Agents.storage_agent as storage_agent
import OpenAi.SuperMario.Agents.visualisation_agent as visualiser





#VISUALISE STATS

episodes = 10000
stats_every = 100

env_name = "SuperMarioBros-v0"
training_version_name = "Simple_NN"


EPISODES_NAME = "env-{}__v-{}__ep-{}__stats-{}__episodes".format(env_name, training_version_name, episodes, stats_every)
REWARDS_NAME = "env-{}__v-{}__ep-{}__stats-{}__rewards".format(env_name, training_version_name, episodes, stats_every)
LABEL = "{}_{}".format(env_name, training_version_name)

visualiser.plot_episodic_rewards(EPISODES_NAME=EPISODES_NAME, REWARDS_NAME=REWARDS_NAME, LABEL=LABEL)
