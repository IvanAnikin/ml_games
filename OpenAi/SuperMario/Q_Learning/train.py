
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT, RIGHT_ONLY
from nes_py.wrappers import JoypadSpace


import OpenAi.SuperMario.Agents.Agent as Agents_file



episodes = 10
# Q Learning Agent Parameters
ALPHA = 0.1          # Learning rate
MIN_EPSILON = 0.05   # Random move probability
GAMMA = 0.95         # Discount factor

num_hidden = 128
show_model = True

seed = 42

env_name = "SuperMarioBros-v0"
training_version_name = "Actor_Critic_1"
movement_type = "SIMPLE_MOVEMENT"

env = gym_super_mario_bros.make(env_name)
env = JoypadSpace(env, SIMPLE_MOVEMENT)
env.seed(seed)                                          # SEED -- ?



Agent = Agents_file.Q_Learning(env=env, gamma=GAMMA, alpha=ALPHA, show_model=show_model, num_hidden=num_hidden)



for episode in range(episodes):
    old_state = state = env.reset()

    while (True):


        action = Agent.act(old_state)

        state, reward, done, info = env.step(action)


        if info['flag_get']:
            print('WE GOT THE FLAG!!!!!!!')
            flag = True

        if done:
            break

        old_state = state