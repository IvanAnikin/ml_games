import gym
import numpy as np
import tensorflow as tf

from statistics import median, mean
from collections import Counter
from tensorflow import keras



import random
import numpy as np
#import tflearn
#from tflearn.layers.core import input_data, dropout, fully_connected
#from tflearn.layers.estimator import regression
from statistics import median, mean
from collections import Counter


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout




print("starting with ml in pycharm")




LR = 1e-3
env = gym.make("CartPole-v0")
env.reset()
goal_steps = 500
score_requirement = 50
initial_games = 10000


def initial_population():
    # [OBS, MOVES]
    training_data = []
    # all scores:
    scores = []
    # just the scores that met our threshold:
    accepted_scores = []
    # iterate through however many games we want:
    for _ in range(initial_games):
        score = 0
        # moves specifically from this environment:
        game_memory = []
        # previous observation that we saw
        prev_observation = []
        # for each frame in 200
        for _ in range(goal_steps):
            # choose random action (0 or 1)
            action = random.randrange(0, 2)
            # do it!
            observation, reward, done, info = env.step(action)

            # notice that the observation is returned FROM the action
            # so we'll store the previous observation here, pairing
            # the prev observation to the action we'll take.
            if len(prev_observation) > 0:
                game_memory.append([prev_observation, action])
            prev_observation = observation
            score += reward
            if done: break

        # IF our score is higher than our threshold, we'd like to save
        # every move we made
        # NOTE the reinforcement methodology here.
        # all we're doing is reinforcing the score, we're not trying
        # to influence the machine in any way as to HOW that score is
        # reached.
        if score >= score_requirement:
            accepted_scores.append(score)
            for data in game_memory:
                # convert to one-hot (this is the output layer for our neural network)
                if data[1] == 1:
                    output = [0, 1]
                elif data[1] == 0:
                    output = [1, 0]

                # saving our training data
                training_data.append([data[0], output])

        # reset env to play again
        env.reset()
        # save overall scores
        scores.append(score)

    # just in case you wanted to reference later
    # training_data_save = np.array(training_data)
    # np.save('saved.npy',training_data_save)

    # some stats here, to further illustrate the neural network magic!
    print()
    print()
    print('Average accepted score:', mean(accepted_scores))
    print('Median score for accepted scores:', median(accepted_scores))
    #print(Counter(accepted_scores))
    print()
    print()

    return training_data



def neural_network_model(input_size):
    model = keras.Sequential()
    model.add(keras.Input(shape=(None, input_size, 1)))
    model.add(keras.layers.Dense(128, activation="relu"))
    model.add(keras.layers.Dense(256, activation="relu"))
    model.add(keras.layers.Dense(128, activation="relu"))
    model.add(keras.layers.Dense(2, activation="softmax"))

    return model


training_data = initial_population()

X = np.array([i[0] for i in training_data]).reshape(-1,len(training_data[0][0]),1)
X2 = np.array([i[0] for i in training_data])

print(X2[0])
print("basic X shape: ", X2.shape)
print("------------")
print(X[0])
print("reshaped X shape: ", X.shape)
print()
print()

y = np.array([i[1] for i in training_data])
print(y[0])
print(y[1])
print(y[2])
print("y shape: ", y.shape)     #--- AttributeError: 'list' object has no attribute 'shape'
print()
print()

model = neural_network_model(input_size = len(X[0]))

model.summary()

#optimizer='adam',
model.compile(optimizer=keras.optimizers.Adam(lr=LR),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

#history = model.fit(X, y, epochs=5)
#data_adapter.py 971, in select_data_adapter
#ValueError: Failed to find data adapter that can handle input: <class 'numpy.ndarray'>, (<class 'list'> containing values of types {'(<class \'list\'> containing values of types {"<class \'int\'>"})'})
# Arguments:
#       x: Input data. It could be:
#         - A Numpy array (or array-like), or a list of arrays
#           (in case the model has multiple inputs).
#         - A TensorFlow tensor, or a list of tensors
#           (in case the model has multiple inputs).
#         - A dict mapping input names to the corresponding array/tensors,
#           if the model has named inputs.
#         - A `tf.data` dataset. Should return a tuple
#           of either `(inputs, targets)` or
#           `(inputs, targets, sample_weights)`.
#         - A generator or `keras.utils.Sequence` returning `(inputs, targets)`
#           or `(inputs, targets, sample_weights)`.
#         A more detailed description of unpacking behavior for iterator types
#         (Dataset, generator, Sequence) is given below.
#       y: Target data. Like the input data `x`,
#         it could be either Numpy array(s) or TensorFlow tensor(s).
#         It should be consistent with `x` (you cannot have Numpy inputs and
#         tensor targets, or inversely). If `x` is a dataset, generator,
#         or `keras.utils.Sequence` instance, `y` should
#         not be specified (since targets will be obtained from `x`).

#ValueError: Shapes (None, 2) and (None, 4, 1, 2) are incompatible
#WARNING:tensorflow:Model was constructed with shape (None, None, 4, 1) for input Tensor("input_1:0", shape=(None, None, 4, 1), dtype=float32), but it was called on an input with incompatible shape (None, 4, 1, 1).


#GAME TESTING
#env = gym.make("CartPole-v0")
#scores = []
#num_trials = 50
#sim_steps = 500
#for _ in range(num_trials):
#    env.reset()
#    game_memory = []
#    observation = []
#    score = 0
#    for step in range(sim_steps):
#        env.render()
#        if len(observation) == 0:
#            action = random.randrange(0,2)
#        else:
#            action = np.argmax(model.predict(observation.reshape(1,4)))
#        observation, reward, done, info = env.step(action)
#        prev_obs = observation
#        game_memory.append([observation, action])
#        score += reward
#        if done:
#            break
#    scores.append(score)
#
#print("Average Score:", np.mean(scores))