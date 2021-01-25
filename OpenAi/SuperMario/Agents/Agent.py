

import tensorflow as tf
from tensorflow import keras
from keras import layers
from keras.optimizers import Adam
from keras.losses import Huber

from typing import Tuple, List
import numpy as np
from collections import deque
import random

import OpenAi.SuperMario.Agents.Models as Models
import OpenAi.SuperMario.Agents.hyperparameters as hp



class DQN_Agent():
    def __init__(self, env, states, num_hidden, epsilon, eps_decay, eps_min,
                 max_memory, copy, learn_each, save_each, batch_size, gamma, show_model, double_q):
        self.env = env
        self.states = states
        self.num_hidden = num_hidden
        self.epsilon = epsilon
        self.eps_decay = eps_decay
        self.eps_min = eps_min
        self.memory = deque(maxlen=max_memory)
        self.copy = copy
        self.batch_size = batch_size
        self.gamma = gamma

        self.step = 0
        self.learn_step = 0
        self.learn_each = learn_each
        self.save_each = save_each
        self.num_actions = env.action_space.n
        self.double_q = double_q                       # DQ - True

        self.model_online = self.generate_model()
        self.model_target = self.generate_model()
        #self.generate_model_2()
        #self.generate_model_2()


        # MODELS VISUALISATION
        if(show_model):
            print("Model 'online' summary: ")
            self.model_online.summary()
            print("Model 'target': same structure")



    def run(self, state):

        state = tf.convert_to_tensor(state)
        state = tf.expand_dims(state, 0)

        if np.random.rand() < self.epsilon:
            # Random action
            action = self.env.action_space.sample()
        else:
            # Policy action
            q = self.model_online(state)
            action = np.argmax(q)

        # Decrease eps
        self.epsilon *= self.eps_decay
        self.epsilon = max(self.eps_min, self.epsilon)
        # Increment step
        self.step += 1

        return action


    def learn(self):

        # Sync target network
        if self.step % self.copy == 0:
            self.copy_model()
        # Checkpoint model
        if self.step % self.save_each == 0:
            self.save_model()
                                                        # Break if burn-in
                                                        #if self.step < self.burnin:
                                                        #    return
        # Break if no training
        if self.learn_step < self.learn_each:
            self.learn_step += 1
            return
        # Sample batch
        if(len(self.memory) < self.batch_size): batch = random.sample(self.memory, len(self.memory))
        else: batch = random.sample(self.memory, self.batch_size)
        state, next_state, action, reward, done = map(np.array, zip(*batch))

        # Get next q values from target network
        next_q = self.model_target(next_state)
        # Calculate discounted future reward
        if self.double_q:
            q = self.model_online(next_state)
            a = np.argmax(q, axis=1)
            target_q = reward + (1. - done) * self.gamma * next_q[np.arange(0, self.batch_size), a]
        else:
            target_q = reward + (1. - done) * self.gamma * np.amax(next_q, axis=1)

        # Update model
        #summary, _ = self.session.run(fetches=[self.summaries, self.train],
        #                              feed_dict={self.input: state,
        #                                         self.q_true: np.array(target_q),
        #                                         self.a_true: np.array(action),
        #                                         self.reward: np.mean(reward)})

        self.a_true = np.array(action)
        self.q_true = np.array(target_q)

        # Optimizer
        # --?-- self.action = tf.argmax(input=self.output, axis=1)

        #self.q_pred = tf.gather_nd(params=self.actions,
        #                           indices=tf.stack([tf.range(tf.shape(self.a_true)[0]), self.a_true], axis=1))

        #self.loss = Huber(self.q_true, self.q_pred)
        #elf.train = Adam(learning_rate=0.00025).minimize(self.loss)

        X = state
        Y = self.q_true

        self.model_online.fit(X, Y, verbose=0) # verbose=0 -- logging none


        # Reset learn step
        self.learn_step = 0
        # Write
        #self.writer.add_summary(summary, self.step)

        return

    def copy_model(self):

        return

    def save_model(self):

        return

    def add(self, experience):

        self.memory.append(experience)

        return

    def generate_model(self):

        inputs = layers.Input(shape=self.states)
        input_float = tf.cast(inputs, tf.float32) / 255.

        conv_1 = layers.Convolution2D(filters=32, kernel_size=8, strides=4, activation="relu")(input_float)
        conv_2 = layers.Convolution2D(filters=32, kernel_size=4, strides=2, activation="relu")(conv_1)
        conv_3 = layers.Convolution2D(filters=32, kernel_size=3, strides=1, activation="relu")(conv_2)

        flatten = layers.Flatten()(conv_3)

        common = layers.Dense(self.num_hidden, activation="relu")(flatten)

        actions = layers.Dense(self.num_actions, activation="linear")(common)

        model = keras.Model(inputs=inputs, outputs=actions)
        model.compile(loss="mse", optimizer=Adam(lr=0.001), metrics=['accuracy'])

        return model

    def generate_model_2(self):

        self.inputs = layers.Input(shape=self.states)
        self.input_float = tf.cast(self.inputs, tf.float32) / 255.

        # ONLINE
        self.conv_1 = layers.Convolution2D(filters=32, kernel_size=8, strides=4, activation="relu")(self.input_float)
        self.conv_2 = layers.Convolution2D(filters=32, kernel_size=4, strides=2, activation="relu")(self.conv_1)
        self.conv_3 = layers.Convolution2D(filters=32, kernel_size=3, strides=1, activation="relu")(self.conv_2)
        self.flatten = layers.Flatten()(self.conv_3)
        self.common = layers.Dense(self.num_hidden, activation="relu")(self.flatten)
        self.actions = layers.Dense(self.num_actions, activation="linear")(self.common)

        # TARGET
        self.conv_1_t = layers.Convolution2D(filters=32, kernel_size=8, strides=4, activation="relu")(self.input_float)
        self.conv_2_t = layers.Convolution2D(filters=32, kernel_size=4, strides=2, activation="relu")(self.conv_1_t)
        self.conv_3_t = layers.Convolution2D(filters=32, kernel_size=3, strides=1, activation="relu")(self.conv_2_t)
        self.flatten_t = layers.Flatten()(self.conv_3_t)
        self.common_t = layers.Dense(self.num_hidden, activation="relu")(self.flatten_t)
        self.actions_t = layers.Dense(self.num_actions, activation="linear")(self.common_t)




class Q_Learning():
    def __init__(self, env, gamma, alpha, show_model, num_hidden):

        self.env = env
        self.gamma = gamma
        self.alpha = alpha

        self.num_states = env.observation_space.shape
        self.num_actions = env.action_space.n

        self.crossed1425Gap = False
        self.crossed594Gap = False

        self.stuck_duration = 0
        self.jumps = 0
        self.prev_x_pos = 0

        #self.DISCRETE_OS_SIZE = [20, 20]
        #self.OS_SIZE = list(env.observation_space.shape) # --- ? Discrete size
        #elf.q_table = np.random.uniform(low=0, high=5, size=(self.OS_SIZE + [env.action_space.n]))
        #print(self.q_table)

        self.ModelClass = Models.DQN(input_shape=self.num_states, num_hidden=num_hidden,
                                     num_actions=self.num_actions, show_model=show_model)
        self.model = self.ModelClass.model


    # https://console.paperspace.com/gcn-team/notebook/pr5ddt1g9
    def act(self, state, info):
        #action = self.env.action_space.sample()                                                                            # random actions with epsilon ??

        action_probs = self.model(state)
        action = np.argmax(action_probs)

        # Get Mario unstuck
        # from https://github.com/gabegrand/Super-Mario-RL/blob/7c0eea5b8a81f06bddfa6bb1036f82bba4b0e806/src/approxQAgent.py#L4
        # other potential: https://github.com/gabegrand/Super-Mario-RL/blob/7c0eea5b8a81f06bddfa6bb1036f82bba4b0e806/src/approxSarsaAgent.py


        return action


    def learn(self, state, reward, next_state, done):

        X = state

        next_state = tf.convert_to_tensor(next_state)
        next_state = tf.expand_dims(next_state, 0)

        old_state_action_probs = self.model(state)
        new_state_action_probs = self.model(next_state)

        old_action = np.argmax(old_state_action_probs)
        new_action = np.argmax(new_state_action_probs)

        Y = old_state_action_probs + (self.alpha * (reward + self.gamma * new_state_action_probs - old_state_action_probs))
        if done : Y = tf.convert_to_tensor(np.full((7,), reward))

        #Y = old_state_action_probs
        #Y[old_action] = old_state_action_probs[old_action] + (self.alpha * (reward + self.gamma * new_state_action_probs[old_action] - old_state_action_probs[old_action]))

        self.model.fit(X, Y, verbose=0) # verbose=0 -- logging none

        return


    def calculate_reward(self, reward, info):
        # GAPS CROSSING REWARD
        '''
        if not self.crossed1425Gap and info['x_pos'] > 1425: # and hp.WORLD == (1, 1) and not
            print("Crossed gap! Reward +500!")
            reward += 500
            self.crossedGap = True
        '''
        if not self.crossed594Gap and info['x_pos'] > 594: # and hp.WORLD == (1, 1) and not
            print("Crossed _594_ gap! Reward +500!")
            reward += 500
            self.crossed594Gap = True

        # DEATH PUNISH
        if info['life'] == 0:
             print("Oh no! Mario died!")
             reward -= hp.DEATH_PENALTY


        return reward

    def stuckForTooLong(self, info):
        if(info != 0):

            if self.prev_x_pos == info['x_pos']:
                self.stuck_duration += 1
            else:
                self.stuck_duration = 0

            if self.stuck_duration > 50:
                print("stuck")
                self.stuck_duration = 0
                return True

            self.prev_x_pos = info['x_pos']

        return False

    # https://github.com/gabegrand/Super-Mario-RL/blob/7c0eea5b8a81f06bddfa6bb1036f82bba4b0e806/src/heuristicAgent.py
    # Returns the vert distance from Mario to the ground, 0 if on ground
    # if no ground below Mario, return number of rows to offscreen
    # Only call if Mario is on screen
    # Norm factor is state.shape[0], which is 13
    def groundVertDistance(self, state, mpos):
        m_row, m_col = mpos

        if m_row < state.shape[0] - 1:
            # get the rows in Mario's column with objects, if any
            col_contents = state[m_row + 1:, m_col]
            obj_vert_dists = np.nonzero(col_contents == 1)

            if obj_vert_dists[0].size == 0:
                return float(state.shape[0] - m_row - 1) / state.shape[0]
            else:
                return float(obj_vert_dists[0][0]) / state.shape[0]
        else:
            return 1.0 / state.shape[0]

    # Returns mario's position as row, col pair
    # Returns None if Mario not on map
    # Always perform None check on return val
    # For functions in this file, use _marioPosition, since functions
    # should only be called if mario is on screen
    def marioPosition(self, state):
        if state is None:
            return None
        rows, cols = np.nonzero(state == 3)
        if rows.size == 0 or cols.size == 0:
            print("WARNING: Mario is off the map")
            return None
        else:
            return rows[0], cols[0]

    # Return whether Mario can move right in his position (1=true)
    # Only call if Mario is on screen
    def canMoveRight(self, state, mpos):
        m_row, m_col = mpos

        if m_col < state.shape[1] - 1:
            if state[m_row, m_col + 1] != 1:
                return 1.0
            return 0.0
        return 1.0


class Actor_Critic_2():
    def __init__(self, env, gamma, learning_rate, num_hidden):
        self.gamma = gamma
        self.learning_rate = learning_rate

        self.huber_loss = keras.losses.Huber()
        self.optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
        self.eps = np.finfo(np.float32).eps.item()

        self.num_inputs = env.observation_space.shape
        self.num_actions = env.action_space.n
        self.num_hidden = num_hidden

        self.ModelClass = Models.Actor_Critic_2(self.num_inputs, self.num_hidden, self.num_actions)
        self.model = self.ModelClass.model

    def train(self, rewards_history, action_probs_history, critic_value_history, tape):
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
        returns = (returns - np.mean(returns)) / (np.std(returns) + self.eps)
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

# from example:
# https://github.com/yingshaoxo/ML/tree/master/12.reinforcement_learning_with_mario_bros
class Agent_Simple_NN:
    def __init__(self, env, model_file_path, load_model = False, show_model = True):

        self.env = env

        self.num_states = self.env.observation_space.shape
        self.num_actions = self.env.action_space.n

        self.models = Models.Simple_NN(env=env, model_file_path=model_file_path, load_model=load_model, show_model=show_model) # learning rate (other params)


    def get_action(self, prev_state, epsilon):

        if np.random.random() > epsilon:
            action = np.argmax(self.models.model.predict(np.expand_dims(prev_state, axis=0)))
        else:
            action = self.env.action_space.sample()

        return action


class Actor_Critic_1():
    def __init__(
            self,
            gamma,
            num_hidden_units,
            env):
        """Initialize."""
        super().__init__()

        self.env = env
        self.gamma = gamma
        self.num_hidden_units = num_hidden_units

        self.num_states = env.observation_space.shape
        self.num_actions = env.action_space.n

        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
        self.huber_loss = tf.keras.losses.Huber(reduction=tf.keras.losses.Reduction.SUM)

        self.model = Models.ActorCritic_1(self.num_actions, self.num_hidden_units)



    def env_step(self, action: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Returns state, reward and done flag given an action."""

        state, reward, done, _ = self.env.step(action)
        return (state.astype(np.float32),
                np.array(reward, np.int32),
                np.array(done, np.int32))

    def tf_env_step(self, action: tf.Tensor) -> List[tf.Tensor]:
        return tf.numpy_function(self.env_step, [action],
                                 [tf.float32, tf.int32, tf.int32])

    def run_episode(
            self,
            initial_state: tf.Tensor,
            model: tf.keras.Model,
            max_steps: int) -> List[tf.Tensor]:
        """Runs a single episode to collect training data."""

        action_probs = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
        values = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
        rewards = tf.TensorArray(dtype=tf.int32, size=0, dynamic_size=True)

        initial_state_shape = initial_state.shape
        state = initial_state

        for t in tf.range(max_steps):
            # Convert state into a batched tensor (batch size = 1)
            state = tf.expand_dims(state, 0)

            # Run the model and to get action probabilities and critic value
            action_logits_t, value = model(state)

            # Sample next action from the action probability distribution
            action = tf.random.categorical(action_logits_t, 1)[0, 0]
            action_probs_t = tf.nn.softmax(action_logits_t)

            # Store critic values
            values = values.write(t, tf.squeeze(value))

            # Store log probability of the action chosen
            action_probs = action_probs.write(t, action_probs_t[0, action])

            # Apply action to the environment to get next state and reward
            state, reward, done = self.tf_env_step(action)
            state.set_shape(initial_state_shape)

            # Store reward
            rewards = rewards.write(t, reward)

            if tf.cast(done, tf.bool):
                break

        action_probs = action_probs.stack()
        values = values.stack()
        rewards = rewards.stack()

        return action_probs, values, rewards

    def get_expected_return(
            self,
            rewards: tf.Tensor,
            gamma: float,
            epsilon: float,
            standardize: bool = True) -> tf.Tensor:
        """Compute expected returns per timestep."""

        n = tf.shape(rewards)[0]
        returns = tf.TensorArray(dtype=tf.float32, size=n)

        # Start from the end of `rewards` and accumulate reward sums
        # into the `returns` array
        rewards = tf.cast(rewards[::-1], dtype=tf.float32)
        discounted_sum = tf.constant(0.0)
        discounted_sum_shape = discounted_sum.shape
        for i in tf.range(n):
            reward = rewards[i]
            discounted_sum = reward + gamma * discounted_sum
            discounted_sum.set_shape(discounted_sum_shape)
            returns = returns.write(i, discounted_sum)
        returns = returns.stack()[::-1]

        if standardize:
            returns = ((returns - tf.math.reduce_mean(returns)) /
                       (tf.math.reduce_std(returns) + epsilon))

        return returns

    def compute_loss(
            self,
            action_probs: tf.Tensor,
            values: tf.Tensor,
            returns: tf.Tensor) -> tf.Tensor:
        """Computes the combined actor-critic loss."""

        advantage = returns - values

        action_log_probs = tf.math.log(action_probs)
        actor_loss = -tf.math.reduce_sum(action_log_probs * advantage)

        critic_loss = self.huber_loss(values, returns)

        return actor_loss + critic_loss

    #@tf.function
    def train_step(
            self,
            initial_state: tf.Tensor,
            #model: tf.keras.Model,
            #optimizer: tf.keras.optimizers.Optimizer,
            #gamma: float,
            max_steps_per_episode: int) -> tf.Tensor:
        """Runs a model training step."""

        with tf.GradientTape() as tape:
            # Run the model for one episode to collect training data
            action_probs, values, rewards = self.run_episode(initial_state, self.model, max_steps_per_episode)
            # data for max_steps
            # action_probs  =   log probability of the action chosen
            # values        =   critic values for actions
            # rewards       =   received reward

            # Calculate expected returns
            returns = self.get_expected_return(rewards, self.gamma)
            # discounted_sum of reward for each episode

            # Convert training data to appropriate TF tensor shapes
            action_probs, values, returns = [
                tf.expand_dims(x, 1) for x in [action_probs, values, returns]]

            # Calculating loss values to update our network
            loss = self.compute_loss(action_probs, values, returns)

        # Compute the gradients from the loss
        grads = tape.gradient(loss, self.model.trainable_variables)

        # Apply the gradients to the model's parameters
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

        episode_reward = tf.math.reduce_sum(rewards)

        return episode_reward







"""
The `Buffer` class implements Experience Replay.
---
![Algorithm](https://i.imgur.com/mS6iGyJ.jpg)
---
**Critic loss** - Mean Squared Error of `y - Q(s, a)`
where `y` is the expected return as seen by the Target network,
and `Q(s, a)` is action value predicted by the Critic network. `y` is a moving target
that the critic model tries to achieve; we make this target
stable by updating the Target model slowly.
**Actor loss** - This is computed using the mean of the value given by the Critic network
for the actions taken by the Actor network. We seek to maximize this quantity.
Hence we update the Actor network so that it produces actions that get
the maximum predicted value as seen by the Critic, for a given state.
"""
class Agent:
    def __init__(self, env, buffer_capacity=100000, batch_size=64, gamma = -0.99, tau = 0.005, critic_lr = 0.002, actor_lr = 0.001):
        num_states = env.observation_space.shape
        num_actions = env.action_space.n
        # Number of "experiences" to store at max
        self.buffer_capacity = buffer_capacity
        # Num of tuples to train on.
        self.batch_size = batch_size

        # Its tells us num of times record() was called.
        self.buffer_counter = 0

        self.gamma = gamma
        self.tau = tau
        self.critic_lr = critic_lr
        self.actor_l = actor_lr

        # Instead of list of tuples as the exp.replay concept go
        # We use different np.arrays for each tuple element
        self.state_buffer = np.zeros((self.buffer_capacity, num_states))
        self.action_buffer = np.zeros((self.buffer_capacity, num_actions))
        self.reward_buffer = np.zeros((self.buffer_capacity, 1))
        self.next_state_buffer = np.zeros((self.buffer_capacity, num_states))


        self.models = Models.Models(env=env, critic_lr = critic_lr, actor_lr = actor_lr)



    # Takes (s,a,r,s') obervation tuple as input
    def record(self, obs_tuple):
        # Set index to zero if buffer_capacity is exceeded,
        # replacing old records
        index = self.buffer_counter % self.buffer_capacity

        self.state_buffer[index] = obs_tuple[0]
        self.action_buffer[index] = obs_tuple[1]
        self.reward_buffer[index] = obs_tuple[2]
        self.next_state_buffer[index] = obs_tuple[3]

        self.buffer_counter += 1

    # Eager execution is turned on by default in TensorFlow 2. Decorating with tf.function allows
    # TensorFlow to build a static graph out of the logic and computations in our function.
    # This provides a large speed up for blocks of code that contain many small TensorFlow operations such as this one.
    @tf.function
    def update(self, state_batch, action_batch, reward_batch, next_state_batch,):
        # Training and updating Actor & Critic networks.
        # See Pseudo Code.
        with tf.GradientTape() as tape:
            target_actions = self.models.target_actor(next_state_batch, training=True)
            y = reward_batch + self.gamma * self.models.target_critic(
                [next_state_batch, target_actions], training=True
            )
            critic_value = self.models.critic_model([state_batch, action_batch], training=True)
            critic_loss = tf.math.reduce_mean(tf.math.square(y - critic_value))

        self.critic_grad = tape.gradient(critic_loss, self.models.critic_model.trainable_variables)
        self.models.critic_optimizer.apply_gradients(
            zip(self.critic_grad, self.models.critic_model.trainable_variables)
        )

        with tf.GradientTape() as tape:
            actions = self.models.actor_model(state_batch, training=True)
            critic_value = self.models.critic_model([state_batch, actions], training=True)
            # Used `-value` as we want to maximize the value given
            # by the critic for our actions
            actor_loss = -tf.math.reduce_mean(critic_value)

        actor_grad = tape.gradient(actor_loss, self.models.actor_model.trainable_variables)
        self.models.actor_optimizer.apply_gradients(
            zip(actor_grad, self.models.actor_model.trainable_variables)
        )

    # We compute the loss and update parameters
    def learn(self):
        # Get sampling range
        record_range = min(self.buffer_counter, self.buffer_capacity)
        # Randomly sample indices
        batch_indices = np.random.choice(record_range, self.batch_size)
        # Convert to tensors
        state_batch = tf.convert_to_tensor(self.state_buffer[batch_indices])
        action_batch = tf.convert_to_tensor(self.action_buffer[batch_indices])
        reward_batch = tf.convert_to_tensor(self.reward_buffer[batch_indices])
        reward_batch = tf.cast(reward_batch, dtype=tf.float32)
        next_state_batch = tf.convert_to_tensor(self.next_state_buffer[batch_indices])

        self.update(state_batch, action_batch, reward_batch, next_state_batch)

        self.update_target(self.models.target_actor.variables, self.models.actor_model.variables, self.tau)
        self.update_target(self.models.target_critic.variables, self.models.critic_model.variables, self.tau)


    def action(self, state, noise_object):
        return self.models.policy(state, noise_object)

    @tf.function
    def update_target(self, target_weights, weights, tau):
        for (a, b) in zip(target_weights, weights):
            a.assign(b * tau + a * (1 - tau))


"""
To implement better exploration by the Actor network, we use noisy perturbations,
specifically
an **Ornstein-Uhlenbeck process** for generating noise, as described in the paper.
It samples noise from a correlated normal distribution.
"""
class OUActionNoise:
    def __init__(self, mean, std_deviation, theta=0.15, dt=1e-2, x_initial=None):
        self.theta = theta
        self.mean = mean
        self.std_dev = std_deviation
        self.dt = dt
        self.x_initial = x_initial
        self.reset()

    def __call__(self):
        # Formula taken from https://www.wikipedia.org/wiki/Ornstein-Uhlenbeck_process.
        x = (
            self.x_prev
            + self.theta * (self.mean - self.x_prev) * self.dt
            + self.std_dev * np.sqrt(self.dt) * np.random.normal(size=self.mean.shape)
        )
        # Store x into x_prev
        # Makes next noise dependent on current one
        self.x_prev = x
        return x

    def reset(self):
        if self.x_initial is not None:
            self.x_prev = self.x_initial
        else:
            self.x_prev = np.zeros_like(self.mean)



