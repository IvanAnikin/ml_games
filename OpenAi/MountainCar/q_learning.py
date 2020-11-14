import gym

import numpy as np




def get_discrete_state(env, state):
    discrete_state = (state - env.observation_space.low)/discrete_os_win_size
    return tuple(discrete_state.astype(np.int))  # we use this tuple to look up the 3 Q values for the available actions in the q-table


def main():
    env = gym.make("MountainCar-v0")

    obs_table_size = 20                                                                                         # Can be changed in FUTURE OPTIMIZING
    LR = 0.1                                                                                                    # Can be changed in FUTURE OPTIMIZING
    discount = 0.95                                                                                             # Can be changed in FUTURE OPTIMIZING
    episodes = 10000                                                                                            #

    Discrete_obs_table_size = [obs_table_size] * len(env.observation_space.high)
    Discrete_obs_value = (env.observation_space.high - env.observation_space.low) / Discrete_obs_table_size

    q_table = np.random.uniform(low=-2, high=0, size=(Discrete_obs_table_size, Discrete_obs_value))             # low and high values -- because of reward = -1 until win - 0




    '''
    env.reset()

    done = False
    while not done:
        action = 0

        new_state, reward, done, _ = env.step(action)

        env.render()
    env.close()
    '''