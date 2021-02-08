
# https://github.com/keras-team/keras-io/blob/master/examples/rl/ipynb/actor_critic_cartpole.ipynb

import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT, RIGHT_ONLY
from nes_py.wrappers import JoypadSpace

import numpy as np
import time

import tensorflow as tf

import OpenAi.Agents.storage_agent as storage_agent
import OpenAi.SuperMario.Agents.Agent as Agents



seed = 42
gamma = 0.99
learning_rate = 0.01
max_episodes = 10000
max_steps_per_episode = 2000
train_every = 10
save_weights_every = 100
save_rewards_every = 100
save_gif = True
save_gif_every = 2
num_hidden = 128

epsilon = 1
eps_decay = 0.999999
eps_min = 0.1

env_name = "SuperMarioBros-v0"
training_version_name = "Actor_Critic_1"
movement_type = "SIMPLE_MOVEMENT"

env = gym_super_mario_bros.make(env_name)
env = JoypadSpace(env, SIMPLE_MOVEMENT)
env.seed(seed)

num_actions = env.action_space.n


Agent = Agents.Actor_Critic_2(env=env, gamma=gamma, learning_rate=learning_rate, num_hidden=num_hidden)


# VISUALISE PARAMETERS
print("env_name: ", env_name)
print("training_version_name: ", training_version_name)
print("movement_type: ", movement_type)
print()
print("max_episodes: ", max_episodes)
print("max_steps_per_episode: ", max_steps_per_episode)
print("gamma: ", gamma)
print("learning_rate: ", learning_rate)
print("num_hidden: ", num_hidden)
Agent.model.summary()
print()



action_probs_history = []
critic_value_history = []
rewards_history = []
running_reward = 0
episode_count = 0

stats_ep_rewards = {'ep': [], 'avg': [], 'x_pos': []}


for episode in range(max_episodes):  # Run until solved
    state = env.reset()
    episode_reward = 0
    frames = []
    current_step = 0
    start = time.time()
    #for timestep in range(1, max_steps_per_episode):
    while(True):
        env.render()
        with tf.GradientTape() as tape:

            state = tf.convert_to_tensor(state)
            state = tf.expand_dims(state, 0)

            # Predict action probabilities and estimated future rewards
            # from environment state
            action_probs, critic_value = Agent.model(state) #model(state)
            critic_value_history.append(critic_value[0, 0])

            # Sample action from action probability distribution
            action = np.random.choice(num_actions, p=np.squeeze(action_probs))

            if np.random.rand() < epsilon:
                # Random action
                action = env.action_space.sample()

            action_probs_history.append(tf.math.log(action_probs[0, action]))


            # Apply the sampled action in our environment
            state, reward, done, info = env.step(action)
            rewards_history.append(reward)
            episode_reward += reward

            # MODEL TRAINING
            if(current_step % train_every == 0 and current_step != 0):

                Agent.train(rewards_history=rewards_history, action_probs_history=action_probs_history, critic_value_history=critic_value_history, tape=tape)

                # Clear the loss and reward history
                action_probs_history.clear()
                critic_value_history.clear()
                rewards_history.clear()

            epsilon *= eps_decay
            epsilon = max(eps_min, epsilon)

            # GIFS SAVING
            #if (save_gif and episode_count % save_gif_every == 0):  # and episode_count != 0
                #env.render()
                #frames.append(env.render(mode="rgb_array"))

        # Update running reward to check condition for solving
        running_reward = 0.05 * episode_reward + (1 - 0.05) * running_reward

        if info['flag_get']:
            print('WE GOT THE FLAG!!!!!!!')
            flag = True

        if done:
            break

        if (current_step > 500 and current_step != 0 and episode_count < 10):
            print("ending episode at step: ", current_step, " | x_pos: ", info['x_pos'])
            break
        if(current_step > 1000 and current_step != 0 and episode_count < 50):
            print("ending episode at step: ", current_step, " | x_pos: ", info['x_pos'])
            break
        if (current_step > 2000 and current_step != 0 and episode_count < 100):
            print("ending episode at step: ", current_step, " | x_pos: ", info['x_pos'])
            break
        if (current_step > 5000 and current_step != 0 and episode_count < 1000):
            print("ending episode at step: ", current_step, " | x_pos: ", info['x_pos'])
            break

        current_step += 1

    # GIFS SAVING
    #if (save_gif and episode_count % save_gif_every == 0):  # and episode_count != 0
        #name = "{}__{}__gif__{}".format(env_name, training_version_name, episode_count)
        #storage_agent.save_frames_as_gif(frames=frames, name=name)
        #display_frames_as_gif(frames)

    episode_count += 1



    # Log details
    stats_ep_rewards['ep'].append(episode_count)
    stats_ep_rewards['avg'].append(episode_reward)
    stats_ep_rewards['x_pos'].append(info['x_pos'])

    # SAVE WEIGHTS
    if (episode_count % save_weights_every == 0):
        model_file_path = "./{}__{}__{}__ac_model__{}.HDF5".format(env_name, training_version_name, movement_type,
                                                                   episode_count)
        Agent.model.save(model_file_path)

    # SAVE REWARDS
    if (episode_count % save_rewards_every == 0):
        EPISODES_NAME = "{}__{}__{}__stats_ep__{}".format(env_name, training_version_name, movement_type, episode_count)
        REWARDS_NAME = "{}__{}__{}__stats_avg__{}".format(env_name, training_version_name, movement_type, episode_count)
        X_POS_NAME = "{}__{}__{}__stats_x_pos__{}".format(env_name, training_version_name, movement_type, episode_count)
        storage_agent.save_np(name=EPISODES_NAME, data=np.array(stats_ep_rewards['ep']))
        storage_agent.save_np(name=REWARDS_NAME, data=np.array(stats_ep_rewards['avg']))
        storage_agent.save_np(name=X_POS_NAME, data=np.array(stats_ep_rewards['x_pos']))

    #if episode_count % 10 == 0:
    template = "running reward: {:.2f} at episode {} || x_pos: {} || steps: {} || episode time: {}"
    print(template.format(running_reward, episode_count, info['x_pos'], current_step, time.time() - start))


EPISODES_NAME = "{}__{}__{}__stats_ep__{}".format(env_name, training_version_name, movement_type, episode_count)
REWARDS_NAME = "{}__{}__{}__stats_avg__{}".format(env_name, training_version_name, movement_type, episode_count)
X_POS_NAME = "{}__{}__{}__stats_x_pos__{}".format(env_name, training_version_name, movement_type, episode_count)
storage_agent.save_np(name=EPISODES_NAME, data=np.array(stats_ep_rewards['ep']))
storage_agent.save_np(name=REWARDS_NAME, data=np.array(stats_ep_rewards['avg']))
storage_agent.save_np(name=X_POS_NAME, data=np.array(stats_ep_rewards['x_pos']))