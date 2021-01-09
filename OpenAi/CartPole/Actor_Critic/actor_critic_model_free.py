#https://github.com/philtabor/Youtube-Code-Repository/tree/master/ReinforcementLearning/PolicyGradient/actor_critic/tensorflow2


import gym
import numpy as np
from OpenAi.CartPole.Actor_Critic.agent import Agent
from OpenAi.CartPole.Actor_Critic.utils import plot_learning_curve

'''

env = gym.make('CartPole-v0')
n_games = 2000
batch_size = 32



ActorCritic = Pendulum_AC_nets.ActorCritic(env=env)



for i in range(n_games):

    observation = env.reset()
    done = False

    while not done:
        #predict action
        action = ActorCritic.choose_action(np.reshape(observation,[1,-1])).reshape((1, env.action_space.shape[0]))
        #act
        observation_, reward, done, info = env.step(action)
        #learn
        ActorCritic.train(batch_size=batch_size)

        observation = observation_

        #VISUALISATION
        env.render()
   '''


'''
num_hidden_units = 128
num_actions = env.action_space.shape[0] #1
ActorCritic = Pendulum_AC_nets.ActorCritic_v2(num_actions=num_actions, num_hidden_units=num_hidden_units)



for i in range(n_games):

    observation = env.reset()
    done = False

    while not done:
        #predict action
        action, value = ActorCritic.call(observation)

        #act
        observation_, reward, done, info = env.step(action)
        #learn
        ActorCritic.train(batch_size=batch_size)

        observation = observation_

        #VISUALISATION
        env.render()
'''




env = gym.make('CartPole-v0')
n_actions = 1
agent = Agent(alpha=1e-5, n_actions=1)
n_games = 1800
# uncomment this line and do a mkdir tmp && mkdir video if you want to
# record video of the agent playing the game.
#env = wrappers.Monitor(env, 'tmp/video', video_callable=lambda episode_id: True, force=True)
filename = 'cartpole_1e-5_1024x512_1800games.png'

figure_file = 'plots/' + filename

best_score = env.reward_range[0]
score_history = []
load_checkpoint = False

if load_checkpoint:
    agent.load_models()

for i in range(n_games):
    observation = env.reset()
    done = False
    score = 0
    while not done:
        action = agent.choose_action(observation)
        observation_, reward, done, info = env.step(action)
        score += reward
        if not load_checkpoint:
            agent.learn(observation, reward, observation_, done) #action,
        observation = observation_
    score_history.append(score)
    avg_score = np.mean(score_history[-100:])

    if avg_score > best_score:
        best_score = avg_score
        if not load_checkpoint:
            agent.save_models()

    if i % 10 == 0:
        #print('episode ', i, 'score %.1f' % score, 'avg_score %.1f' % avg_score)
        print('episode ', i, 'avg_score %.1f' % avg_score)

if not load_checkpoint:
    x = [i+1 for i in range(n_games)]
    plot_learning_curve(x, score_history, figure_file)
