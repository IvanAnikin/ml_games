#https://github.com/keras-team/keras-io/blob/master/examples/rl/ipynb/actor_critic_cartpole.ipynb



import gym
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

import OpenAi.Agents.storage_agent as storage_agent


seed = 42
gamma = 0.99
max_episodes = 400
max_steps_per_episode = 1000
save_gif = True
save_gif_every = 25
stats_every = 10
env_name = "CartPole-v0"
training_version_name = "Actor_Critic_1"

env = gym.make(env_name)
env.seed(seed)

num_inputs = env.observation_space.shape[0]
num_actions = env.action_space.n
num_hidden = 128



eps = np.finfo(np.float32).eps.item()



inputs = layers.Input(shape=(num_inputs,))
common = layers.Dense(num_hidden, activation="relu")(inputs)
action = layers.Dense(num_actions, activation="softmax")(common)
critic = layers.Dense(1)(common)

model = keras.Model(inputs=inputs, outputs=[action, critic])

model.summary()
print()

"""
## Train
"""

def train():

    # Calculate expected value from rewards
    # - At each timestep what was the total reward received after that timestep
    # - Rewards in the past are discounted by multiplying them with gamma
    # - These are the labels for our critic

    returns = []
    discounted_sum = 0
    for r in rewards_history[::-1]:
        discounted_sum = r + gamma * discounted_sum
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
            huber_loss(tf.expand_dims(value, 0), tf.expand_dims(ret, 0))
        )

    # Backpropagation
    loss_value = sum(actor_losses) + sum(critic_losses)
    grads = tape.gradient(loss_value, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))



optimizer = keras.optimizers.Adam(learning_rate=0.01)
huber_loss = keras.losses.Huber()

action_probs_history = []
critic_value_history = []
rewards_history = []
running_reward = 0
episode_count = 0
avg_score_episodic = 0

stats_ep_rewards = {'ep': [], 'avg': []}

for episode in range(max_episodes):  # Run until solved
    state = env.reset()
    episode_reward = 0
    frames = []
    with tf.GradientTape() as tape:
        for timestep in range(1, max_steps_per_episode):
            # env.render(); Adding this line would show the attempts
            # of the agent in a pop up window.

            state = tf.convert_to_tensor(state)
            state = tf.expand_dims(state, 0)

            # Predict action probabilities and estimated future rewards
            # from environment state
            action_probs, critic_value = model(state)
            critic_value_history.append(critic_value[0, 0])

            # Sample action from action probability distribution
            action = np.random.choice(num_actions, p=np.squeeze(action_probs))
            action_probs_history.append(tf.math.log(action_probs[0, action]))

            # Apply the sampled action in our environment
            state, reward, done, _ = env.step(action)
            rewards_history.append(reward)
            episode_reward += reward

            # GIFS SAVING
            if(save_gif and episode_count % save_gif_every == 0): #and episode_count != 0
                #env.render()
                frames.append(env.render(mode="rgb_array"))

            if done:
                break

        # Update running reward to check condition for solving
        running_reward = 0.05 * episode_reward + (1 - 0.05) * running_reward


        train()

        # Clear the loss and reward history
        action_probs_history.clear()
        critic_value_history.clear()
        rewards_history.clear()


    # GIFS SAVING
    if (save_gif and episode_count % save_gif_every == 0): #and episode_count != 0
        name = "{}__{}__gif__{}".format(env_name, training_version_name, episode_count)
        #storage_agent.save_frames_as_gif(frames=frames, name=name)

    avg_score_episodic = (avg_score_episodic + episode_reward) / 2
    episode_count += 1

    # Log details
    if episode_count % 10 == 0:
        template = "running reward: {:.2f} at episode {}"
        print(template.format(running_reward, episode_count))
        print("avg_score_episodic: {}".format(avg_score_episodic))

    if episode_count % stats_every == 0:
        stats_ep_rewards['ep'].append(episode_count)
        stats_ep_rewards['avg'].append(avg_score_episodic)
        avg_score_episodic = 0


    #if running_reward > 195:  # Condition to consider the task solved
    #    print("Solved at episode {}!".format(episode_count))
    #    break

EPISODES_NAME = "{}__{}__stats_ep__{}".format(env_name, training_version_name, episode_count)
REWARDS_NAME = "{}__{}__stats_avg__{}".format(env_name, training_version_name, episode_count)
storage_agent.save_np(name=EPISODES_NAME, data=np.array(stats_ep_rewards['ep']))
storage_agent.save_np(name=REWARDS_NAME, data=np.array(stats_ep_rewards['avg']))

