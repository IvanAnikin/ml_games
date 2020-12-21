
import OpenAi.MountainCar.Q_Learning.storage_agent as MountainCar_Q_Learning_storage_agent

import gym
import numpy as np

import time
import datetime



def mountain_car_single_game(EPISODES = 25000, LEARNING_RATE = 0.1, epsilon = 0.5, end_epsilon_decaying_position = 2, DISCOUNT = 0.95, DISCRETE_OS_SIZE = [20, 20], SHOW_EVERY = 3000, STATS_EVERY=50):

    env = gym.make("MountainCar-v0")


    discrete_os_win_size = (env.observation_space.high - env.observation_space.low) / DISCRETE_OS_SIZE


    START_EPSILON_DECAYING = 1
    END_EPSILON_DECAYING = EPISODES // end_epsilon_decaying_position
    epsilon_decay_value = epsilon / (END_EPSILON_DECAYING - START_EPSILON_DECAYING)

    q_table = np.random.uniform(low=-2, high=0, size=(DISCRETE_OS_SIZE + [env.action_space.n]))



    successful_episodes = 0

    rewards = []
    stats_ep_rewards = {'ep': [], 'avg': []}

    def get_discrete_state(state):
        discrete_state = (state - env.observation_space.low) / discrete_os_win_size
        return tuple(discrete_state.astype(
            np.int))  # we use this tuple to look up the 3 Q values for the available actions in the q-table

    for episode in range(EPISODES):
        discrete_state = get_discrete_state(env.reset())
        done = False

        episode_reward = 0

        if episode % SHOW_EVERY == 0 and episode != 0:
            print(f"episode: {episode} ||  successful episodes: {successful_episodes}")
            successful_episodes = 0

        while not done:

            if np.random.random() > epsilon:
                # Get action from Q table
                action = np.argmax(q_table[discrete_state])
            else:
                # Get random action
                action = np.random.randint(0, env.action_space.n)

            new_state, reward, done, _ = env.step(action)
            episode_reward += reward

            new_discrete_state = get_discrete_state(new_state)

            #if episode % SHOW_EVERY == 0 and episode != 0:
            #    env.render()

            # If simulation did not end yet after last step - update Q table
            if not done:

                # Maximum possible Q value in next step (for new state)
                max_future_q = np.max(q_table[new_discrete_state])

                # Current Q value (for current state and performed action)
                current_q = q_table[discrete_state + (action,)]

                # And here's our equation for a new Q value for current state and action
                new_q = (1 - LEARNING_RATE) * current_q + LEARNING_RATE * (reward + DISCOUNT * max_future_q)

                # Update Q table with new Q value
                q_table[discrete_state + (action,)] = new_q

            # Simulation ended (for any reson) - if goal position is achived - update Q value with reward directly
            elif new_state[0] >= env.goal_position:
                q_table[discrete_state + (action,)] = 0

                successful_episodes += 1

            discrete_state = new_discrete_state

        rewards.append(episode_reward)
        if not episode % STATS_EVERY:
            stats_ep_rewards['ep'].append(episode)
            stats_ep_rewards['avg'].append(sum(rewards[-STATS_EVERY:])/STATS_EVERY)

        # Decaying is being done every episode if episode number is within decaying range
        if END_EPSILON_DECAYING >= episode >= START_EPSILON_DECAYING:
            epsilon -= epsilon_decay_value


    env.close()

    return stats_ep_rewards, q_table


def save_games(LEARNING_RATES = [0.15, 0.20], EPSILONS = [0.5], END_EPSILON_DECAYING_POSITIONS = [2.0], DISCOUNTS = [0.95], DISCRETE_OS_SIZES = [20], episodes = 5000, show_every = 4000, stats_every = 100):

	games_count = len(LEARNING_RATES) * len(EPSILONS) * len(END_EPSILON_DECAYING_POSITIONS) * len(DISCOUNTS) * len(DISCRETE_OS_SIZES)
	total_rounds_time = 0
	round = 0
	
	print("games_count: ", games_count)
	
	for learning_rate_cycle in range(len(LEARNING_RATES)):
		for epsilon_cycle in range(len(EPSILONS)):
			for end_epsilon_decaying_cycle in range(len(END_EPSILON_DECAYING_POSITIONS)):
				for discount_cycle in range(len(DISCOUNTS)):
					for discrete_os_size_cycle in range(len(DISCRETE_OS_SIZES)):
					
						round += 1
						
						learning_rate = LEARNING_RATES[learning_rate_cycle]
						epsilon = EPSILONS[epsilon_cycle]
						end_epsilon_decaying = END_EPSILON_DECAYING_POSITIONS[end_epsilon_decaying_cycle]
						discount = DISCOUNTS[discount_cycle]
						discrete_os_size = [DISCRETE_OS_SIZES[discrete_os_size_cycle], DISCRETE_OS_SIZES[discrete_os_size_cycle]]
						# '''
						NAME = "ep-{}__stats-{}__lr-{}__eps-{}__epsDec-{}__disc-{}__size-{}".format(episodes, stats_every, learning_rate, epsilon, end_epsilon_decaying, discount, discrete_os_size)
						print(NAME)
						start = time.time()
						stats_ep_rewards = mountain_car_single_game(LEARNING_RATE=learning_rate,
																	epsilon=epsilon,
																	end_epsilon_decaying_position=end_epsilon_decaying,
																	DISCOUNT=discount,
																	DISCRETE_OS_SIZE=discrete_os_size,
																	EPISODES=episodes,
																	SHOW_EVERY=show_every,
																	STATS_EVERY=stats_every)
						round_time = time.time()-start
						total_rounds_time += round_time											
						print("round: |", round, "/", games_count, "|")
						print("round time length: ", time.strftime('%M:%S',round_time), "  |||  time left expected: ", time.strftime('%H:%M:%S', total_rounds_time / round * (games_count - round)))
						stats_ep_rewards_ep = stats_ep_rewards['ep']
						stats_ep_rewards_avg = stats_ep_rewards['avg']

						MountainCar_Q_Learning_storage_agent.save_np(name=NAME, data=np.array(stats_ep_rewards_avg))
						

	EPISODES_NAME = "ep-{}__stats-{}__episodes".format(episodes, stats_every)
	#MountainCar_Q_Learning_storage_agent.save_np(name=EPISODES_NAME, data=np.array(stats_ep_rewards_ep))





#successful episodes - count

#get discrete_os_size from q_table len()

def play_with_given_q_table(q_table, DISCRETE_OS_SIZE, EPISODES, SHOW_EVERY):

    env = gym.make("MountainCar-v0")

    discrete_os_win_size = (env.observation_space.high - env.observation_space.low) / DISCRETE_OS_SIZE
    successful_episodes = 0
    all_scores = 0

    def get_discrete_state(state):
        discrete_state = (state - env.observation_space.low) / discrete_os_win_size
        return tuple(discrete_state.astype(
            np.int))  # we use this tuple to look up the 3 Q values for the available actions in the q-table

    for episode in range(EPISODES):
        discrete_state = get_discrete_state(env.reset())
        done = False

        episode_reward = 0

        #if episode % SHOW_EVERY == 0 and episode != 0:
            #print(f"episode: {episode} ||  successful episodes: {successful_episodes}")
            #successful_episodes = 0

        while not done:

            action = np.argmax(q_table[discrete_state])

            new_state, reward, done, _ = env.step(action)
            episode_reward += reward

            new_discrete_state = get_discrete_state(new_state)

            discrete_state = new_discrete_state

            if episode % SHOW_EVERY == 0 and episode != 0:
                env.render()
            if done:
                all_scores += episode_reward
                if new_state[0] >= env.goal_position:
                    successful_episodes += 1

        
        
        average_score = all_scores/(episode+1)
        #print("episode_reward: ", episode_reward)
        #print("average_score: ", average_score)

    env.close()

    return successful_episodes, average_score