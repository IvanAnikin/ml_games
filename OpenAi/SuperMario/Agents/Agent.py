

import tensorflow as tf
from tensorflow import keras

from typing import Tuple, List
import numpy as np

import OpenAi.SuperMario.Agents.Models as Models



class Q_Learning():
    def __init__(self, env, gamma, alpha, show_model, num_hidden):

        self.env = env
        self.gamma = gamma
        self.alpha = alpha

        self.num_states = env.observation_space.shape
        self.num_actions = env.action_space.n


        #self.DISCRETE_OS_SIZE = [20, 20]
        #self.OS_SIZE = list(env.observation_space.shape) # --- ? Discrete size
        #elf.q_table = np.random.uniform(low=0, high=5, size=(self.OS_SIZE + [env.action_space.n]))
        #print(self.q_table)

        self.ModelClass = Models.DQN(input_shape=self.num_states, num_hidden=num_hidden,
                                     num_actions=self.num_actions, show_model=show_model)
        self.model = self.ModelClass.model


# https://console.paperspace.com/gcn-team/notebook/pr5ddt1g9
    def act(self, state):
        #action = np.argmax(self.q_table[state])

        #action = self.env.action_space.sample()                                                                            # random actions with epsilon ??

        action = self.model(state)


        return action

    def learn(self, state, action_id, reward, next_state, done):

        #if done:
        #    target = reward
        #else:
        #    target = reward + self.gamma * max(self.q_table[next_state])

        #td_error = target - self.q_table[state, action_id]
        #self.q_table[state, action_id] = self.q_table[state, action_id] + self.alpha * td_error

        return





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



