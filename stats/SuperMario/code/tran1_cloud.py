

import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT, RIGHT_ONLY
from nes_py.wrappers import JoypadSpace

import numpy as np

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

#import OpenAi.Agents.storage_agent as storage_agent
#import OpenAi.SuperMario.Agents.Agent as Agents



# https://github.com/keras-team/keras-io/blob/master/examples/rl/ipynb/actor_critic_cartpole.ipynb

def save_np(name, data):

    np.save(name, data)

    load_name = name + ".npy"
    print("saved successfully, first array element: ", np.load(load_name)[1])

class Actor_Critic_2_model():
    def __init__(self, input_shape, num_hidden, num_actions):
        inputs = layers.Input(shape=input_shape)
        flatten = layers.Flatten()(inputs)
        common = layers.Dense(num_hidden, activation="relu")(flatten)

        # common = layers.Dense(num_hidden, activation="relu")(inputs)
        action = layers.Dense(num_actions, activation="softmax")(common)
        critic = layers.Dense(1)(common)

        self.model = keras.Model(inputs=inputs, outputs=[action, critic])

class Actor_Critic_2():
    def __init__(self, env, gamma, learning_rate, num_hidden):
        self.gamma = gamma
        self.learning_rate = learning_rate

        self.huber_loss = keras.losses.Huber()
        self.optimizer = keras.optimizers.Adam(learning_rate=learning_rate)

        self.num_inputs = env.observation_space.shape
        self.num_actions = env.action_space.n
        self.num_hidden = num_hidden

        self.ModelClass = Actor_Critic_2_model(self.num_inputs, self.num_hidden, self.num_actions)
        self.model = self.ModelClass.model

    def train(self, rewards_history, action_probs_history, critic_value_history, eps, tape):
        # Calculate expected value from rewards
        # - At each timestep what was the total reward received after that timestep
        # - Rewards in the past are discounted by multiplying them with gamma
        # - These are the labels for our critic

        returns = []
        discounted_sum = 0
        for r in rewards_history[::-1]:
            discounted_sum = r + self.gamma * discounted_sum
            returns.insert(0, discounted_sum)

        # Normalize
        returns = np.array(returns)
        returns = (returns - np.mean(returns)) / (np.std(returns) + eps)
        returns = returns.tolist()

        # Calculating loss values to update our network
        history = zip(action_probs_history, critic_value_history, returns)
        actor_losses = []
        critic_losses = []

        for log_prob, value, ret in history:
            # At this point in history, the critic estimated that we would get a
            # total reward = `value` in the future. We took an action with log probability
            # of `log_prob` and ended up recieving a total reward = `ret`.
            # The actor must be updated so that it predicts an action that leads to
            # high rewards (compared to critic's estimate) with high probability.
            diff = ret - value
            actor_losses.append(-log_prob * diff)  # actor loss

            # The critic must be updated so that it predicts a better estimate of
            # the future rewards.
            critic_losses.append(
                self.huber_loss(tf.expand_dims(value, 0), tf.expand_dims(ret, 0))
            )
        # Backpropagation
        loss_value = sum(actor_losses) + sum(critic_losses)

        grads = tape.gradient(loss_value, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))



seed = 42
gamma = 0.99
learning_rate = 0.01
max_episodes = 500
max_steps_per_episode = 2000
train_every = 10
save_weights_every = 100
save_rewards_every = 100
save_gif = True
save_gif_every = 2

env_name = "SuperMarioBros-v0"
training_version_name = "Actor_Critic_1"
movement_type = "RIGHT_ONLY"

env = gym_super_mario_bros.make(env_name)
env = JoypadSpace(env, RIGHT_ONLY)
env.seed(seed)

num_inputs = env.observation_space.shape
num_inputs2 = env.observation_space.shape[0]
num_inputs3 = (240, 256, 3)
num_actions = env.action_space.n
num_hidden = 128


Agent = Actor_Critic_2(env=env, gamma=gamma, learning_rate=learning_rate, num_hidden=num_hidden)


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

    for timestep in range(1, max_steps_per_episode):
        # env.render(); Adding this line would show the attempts
        # of the agent in a pop up window.
        with tf.GradientTape() as tape:
            state = tf.convert_to_tensor(state)
            state = tf.expand_dims(state, 0)

            # Predict action probabilities and estimated future rewards
            # from environment state
            action_probs, critic_value = Agent.model(state)
            critic_value_history.append(critic_value[0, 0])

            # Sample action from action probability distribution
            action = np.random.choice(num_actions, p=np.squeeze(action_probs))
            action_probs_history.append(tf.math.log(action_probs[0, action]))

            # Apply the sampled action in our environment
            state, reward, done, info = env.step(action)
            rewards_history.append(reward)
            episode_reward += reward

            # MODEL TRAINING
            if(timestep % train_every == 0 and timestep != 0):

                #train()
                Agent.train(rewards_history=rewards_history, action_probs_history=action_probs_history, critic_value_history=critic_value_history, tape=tape)

                # Clear the loss and reward history
                action_probs_history.clear()
                critic_value_history.clear()
                rewards_history.clear()

            # GIFS SAVING
            #if (save_gif and episode_count % save_gif_every == 0):  # and episode_count != 0
                #env.render()
                #frames.append(env.render(mode="rgb_array"))

        if info['flag_get']:
            print('WE GOT THE FLAG!!!!!!!')
            flag = True

        if done:
            break

        # Update running reward to check condition for solving
        running_reward = 0.05 * episode_reward + (1 - 0.05) * running_reward

    # GIFS SAVING
    #if (save_gif and episode_count % save_gif_every == 0):  # and episode_count != 0
        #name = "{}__{}__gif__{}".format(env_name, training_version_name, episode_count)
        #storage_agent.save_frames_as_gif(frames=frames, name=name)
        #display_frames_as_gif(frames)

    episode_count += 1

    # SAVE WEIGHTS
    if (episode_count % save_weights_every == 0):
        model_file_path = "./{}__{}__{}__ac_model__{}.HDF5".format(env_name, training_version_name, movement_type, episode_count)
        Agent.model.save(model_file_path)

    # SAVE REWARDS
    if (episode_count % save_rewards_every == 0):
        EPISODES_NAME = "{}__{}__{}__stats_ep__{}".format(env_name, training_version_name, movement_type, episode_count)
        REWARDS_NAME = "{}__{}__{}__stats_avg__{}".format(env_name, training_version_name, movement_type, episode_count)
        X_POS_NAME = "{}__{}__{}__stats_x_pos__{}".format(env_name, training_version_name, movement_type, episode_count)
        save_np(name=EPISODES_NAME, data=np.array(stats_ep_rewards['ep']))
        save_np(name=REWARDS_NAME, data=np.array(stats_ep_rewards['avg']))
        save_np(name=X_POS_NAME, data=np.array(stats_ep_rewards['x_pos']))

    # Log details
    stats_ep_rewards['ep'].append(episode_count)
    stats_ep_rewards['avg'].append(episode_reward)
    stats_ep_rewards['x_pos'].append(info['x_pos'])
    #if episode_count % 10 == 0:
    template = "running reward: {:.2f} at episode {} || x_pos: {}"
    print(template.format(running_reward, episode_count, info['x_pos']))
    