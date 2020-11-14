import time
import numpy as np
import random

import gym


from tensorflow import keras

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout


def prep():

    #for _ in range(10):
    #    print(np.random.uniform(-2,2))

    env = gym.make("Pendulum-v0")

    env.reset()
    for step in range(100):
        env.render()

        observation, reward, done, info = env.step([np.random.uniform(-2,2)])

        print("observation: ", observation)
        print("reward:", reward)
        time.sleep(1)

        if done:
            break


def create_model(LR = 1e-3, dropout = 0.4):
    model = Sequential()
    model.add(Dense(128, input_shape=(3,), activation="relu"))
    model.add(Dropout(dropout))

    model.add(Dense(256, activation="relu"))
    model.add(Dropout(dropout))

    model.add(Dense(512, activation="relu"))
    model.add(Dropout(dropout))

    model.add(Dense(256, activation="relu"))
    model.add(Dropout(dropout))

    model.add(Dense(128, activation="relu"))
    model.add(Dropout(dropout))
    model.add(Dense(1, activation="softmax"))

    model.compile(
        loss="categorical_crossentropy",
        optimizer=keras.optimizers.Adam(lr=LR),                                           #define LR !!!!!!!!!
        metrics=["accuracy"])
    return model


def create_model_small(LR = 1e-3, dropout = 0.4):
    model = Sequential()
    model.add(Dense(128, input_shape=(3,), activation="relu"))
    model.add(Dropout(dropout))

    model.add(Dense(256, activation="relu"))
    model.add(Dropout(dropout))

    model.add(Dense(128, activation="relu"))
    model.add(Dropout(dropout))
    model.add(Dense(1, activation="softmax"))

    model.compile(
        loss="categorical_crossentropy",
        optimizer=keras.optimizers.Adam(lr=LR),  # define LR !!!!!!!!!
        metrics=["accuracy"])
    return model


def all_games(num_trials = 1000, sim_steps = 199):
    env = gym.make("Pendulum-v0")


    memory = []
    SHOW_EVERY = 500

    for episode in range(num_trials):
        observation = env.reset()

        game_memory = []

        for step in range(sim_steps):

            action = np.random.uniform(-2, 2)

            observation, reward, done, wtff = env.step([action])

            if (done):
                break

            game_memory.append([observation, action, reward])

        memory.append(game_memory)

        if(episode % SHOW_EVERY == 0): print("episode: ", episode, " ||  memo len:", len(memory), " || first step: ", memory[episode][0])

    return memory

def save_games():
    np.save("memory_all_games_10000", all_games(10000))

    print(np.load("memory_all_games_10000.npy")[0][0])



def test_games(num_trials = 1000, sim_steps = 199):

    env = gym.make("Pendulum-v0")

    max_reward = -10000
    least_velocity = 10000

    velocities = []

    for _ in range(num_trials):
        observation = env.reset()

        trial_reward = 0
        trial_velocity = 0


        for step in range(sim_steps):


            action = np.random.uniform(-2,2)

            observation, reward, done, wtff = env.step([action])


            trial_velocity += abs(observation[2])
            trial_reward += reward

            if(done):
                break

        velocities.append(trial_velocity)

        if (trial_reward > max_reward):
            max_reward = trial_reward

        if (trial_velocity < least_velocity):
            least_velocity = trial_velocity



    #print("max_reward: ", max_reward)
    print("Average velocity: {}".format(np.mean(velocities)))
    print("Median velocity: {}".format(np.median(velocities)))


def training_data(num_trials = 1000, min_score = -900, min_velocity = 700, sim_steps = 199):
    env = gym.make("Pendulum-v0")

    print("num_trials: ", num_trials, "|| min_score: ", min_score, "|| min_velocity: ", min_velocity)

    trainingX, trainingY = [], []
    max_reward = -10000

    scores = []
    velocities = []
    SHOW_EVERY = 200

    for episode in range(num_trials):
        observation = env.reset()

        trial_reward = 0
        trial_velocity = 0
        game_memory = []
        training_sampleX, training_sampleY = [], []

        for step in range(sim_steps):

            action = np.random.uniform(-2, 2)                       # --> From NN

            observation, reward, done, wtff = env.step([action])
            game_memory.append([observation, action])
            trial_reward += reward
            trial_velocity += abs(observation[2])

            training_sampleX.append(observation)
            training_sampleY.append(action)

            if(done):
                print("WON")
                break

        if (trial_reward > max_reward): max_reward = trial_reward

        if (trial_reward > min_score and trial_velocity < min_velocity):

            trainingX += training_sampleX
            trainingY += training_sampleY

            scores.append(trial_reward)
            velocities.append(trial_velocity)


        if episode % SHOW_EVERY == 0: print("episode: ", episode, "saved steps: ", len(trainingY))

    #print(len(trainingY))
    #print("trainingX[0]): ", trainingX[0])
    #print("trainingY[0]): ", trainingY[0])

    trainingX, trainingY = np.array(trainingX), np.array(trainingY)
    print("Saved steps: ", trainingX.shape)
    print("Best score: ", max_reward)
    print("Average score: {}".format(np.mean(scores)))
    print("Median score: {}".format(np.median(scores)))
    print("Average velocity: {}".format(np.mean(velocities)))
    print("Median velocity: {}".format(np.median(velocities)))


    return trainingX, trainingY


def training_data_by_step(num_trials = 1000, min_score = -1000, sim_steps = 199):
    env = gym.make("Pendulum-v0")

    print("num_trials: ", num_trials, " ||  min_score: ", min_score)
    print()

    trainingX, trainingY = [], []
    max_reward = -10000

    scores = []
    velocities = []
    SHOW_EVERY = 200

    saved_by_velocity = []
    saved_by_reward = []
    saved_by_both = []

    for episode in range(num_trials):
        observation = env.reset()

        trial_reward = 0
        trial_velocity = 0

        prev_velocity = 0
        prev_reward = 0

        training_sampleX, training_sampleY = [], []

        for step in range(sim_steps):

            action = np.random.uniform(-2, 2)

            observation, reward, done, wtff = env.step([action])

            trial_reward += reward
            trial_velocity += abs(observation[2])

            """
            if(prev_velocity - observation[2] > 0.5):
                saved_by_velocity.append([observation, reward])
                velocities.append(trial_velocity)


            if(reward - prev_reward > 0):
                saved_by_reward.append([observation, reward])
                scores.append(trial_reward)
                
            """

            if(prev_velocity - observation[2] > 0.5 and reward - prev_reward > 0):

                training_sampleX.append(observation)
                training_sampleY.append(action)


            prev_velocity = observation[2]
            prev_reward = reward



            if(done):
                print("WON")
                break


        if (trial_reward > max_reward): max_reward = trial_reward

        if (trial_reward > min_score):                                  # ???  and trial_velocity < min_velocity

            trainingX += training_sampleX
            trainingY += training_sampleY

            scores.append(trial_reward)
            velocities.append(trial_velocity)


        if episode % SHOW_EVERY == 0: print("episode: ", episode)
        #if episode % SHOW_EVERY == 0: print("           ", "len(saved_by_velocity): ", len(saved_by_velocity))
        #if episode % SHOW_EVERY == 0: print("           ", "len(saved_by_reward): ", len(saved_by_reward))
        if episode % SHOW_EVERY == 0: print("           ", "len(saved_by_both): ", len(trainingX))


    print()
    print()
    """
    print("Best score: ", max_reward)
    print("Average score: {}".format(np.mean(scores)))
    print("Median score: {}".format(np.median(scores)))
    print("Average velocity: {}".format(np.mean(velocities)))
    print("Median velocity: {}".format(np.median(velocities)))
    print()

    print("len(saved_by_velocity): ", len(saved_by_velocity))
    print("len(saved_by_reward): ", len(saved_by_reward))
    
    """

    trainingX, trainingY = np.array(trainingX), np.array(trainingY)
    print("Saved steps: ", trainingX.shape)

    return trainingX, trainingY



def array_save_test():

    X, Y = training_data_highobs_count(4000)

    np.save("highobs_4000_80_900_1.2_x", X)
    np.save("highobs_4000_80_900_1.2_y", Y)

    print(np.load("highobs_4000_80_900_1.2_x.npy")[0])


def training_data_highobs_count(num_trials = 1000, min_highobscount = 80, min_velocity = 900, height = -0.8, sim_steps = 199):
    env = gym.make("Pendulum-v0")

    print("num_trials: ", num_trials, "|| min_highobscount: ", min_highobscount, "|| height: ", height)

    trainingX, trainingY = [], []
    max_reward = -10000

    scores = []
    velocities = []
    SHOW_EVERY = 200

    highobs_count = [] #* 4
    accepted_highobs_count = []

    max_highobscount = 0

    for episode in range(num_trials):
        observation = env.reset()

        trial_reward = 0
        trial_velocity = 0
        game_memory = []
        training_sampleX, training_sampleY = [], []

        highobs_trial_count = 0   #[0]  * 4

        for step in range(sim_steps):

            action = np.random.uniform(-2, 2)                       # --> From NN

            observation, reward, done, wtff = env.step([action])
            game_memory.append([observation, action])
            trial_reward += reward
            trial_velocity += abs(observation[2])

            training_sampleX.append(observation)
            training_sampleY.append(action)


            if(reward > height):  highobs_trial_count += 1


            if(done):
                print("WON")
                break

        if (trial_reward > max_reward): max_reward = trial_reward

        if (highobs_trial_count > max_highobscount): max_highobscount = highobs_trial_count

        highobs_count.append(highobs_trial_count)

        if(highobs_trial_count > min_highobscount and trial_velocity < min_velocity):
            trainingX += training_sampleX
            trainingY += training_sampleY

            scores.append(trial_reward)
            velocities.append(trial_velocity)
            accepted_highobs_count.append((highobs_trial_count))


        if episode % SHOW_EVERY == 0: print("episode: ", episode, "saved steps: ", len(trainingY))


    trainingX, trainingY = np.array(trainingX), np.array(trainingY)
    print("Saved steps: ", trainingX.shape)
    print("Best score: ", max_reward)
    print("Average score: {}".format(np.mean(scores)))
    print("Median score: {}".format(np.median(scores)))
    print("Average velocity: {}".format(np.mean(velocities)))
    print("Median velocity: {}".format(np.median(velocities)))

    print()
    print()

    #for i in range(len(highobs_count)):
        #print()
        #print(i)
    print("Best highobs: ", max_highobscount)
    #print("Accepted high obs count: ", len(accepted_highobs_count))
    print("Average high obs: ", np.mean(highobs_count)) #[i]
    print("Average accepted high obs: ", np.mean(accepted_highobs_count))


    return trainingX, trainingY
    #return highobs_count




def plot_steps_info():

    game_memory = [] # [0 = velocity_diff, 1 = reward_diff,]

    prev_velocity = 0
    prev_reward = 0

    env = gym.make("Pendulum-v0")

    env.reset()

    for step in range(199):

        env.render()

        action = np.random.uniform(-2, 2)

        observation, reward, done, wtff = env.step([action])

        game_memory.append([abs(prev_velocity-observation[2]), abs(prev_reward - reward)])

        prev_velocity = observation[2]
        prev_reward = reward

        if (done):
            print("WON")
            break

    env.close()

    print("Average velocity_diff: {}".format(np.mean(game_memory[0])))
    print("Median velocity_diff: {}".format(np.median(game_memory[0])))
    print("Average reward_diff: {}".format(np.mean(game_memory[1])))
    print("Median reward_diff: {}".format(np.median(game_memory[1])))



def play_and_train(num_trials = 100, min_score = -900, min_velocity = 700, sim_steps = 199):


    env = gym.make("Pendulum-v0")


    # LOAD TRAINING DATA
    trainingX = np.load("4000_800_700_x.npy")
    trainingY = np.load("4000_800_700_y.npy")

    print(trainingX.shape)

    # MODEL CREATION
    model = create_model_small(1e-3, 0.4)

    model.summary()
    print()

    # MODEL FITTING TRAINING DATA
    epochs = 10
    history = model.fit(trainingX, trainingY, epochs=epochs)



    # PLAYING -- DECISIONS FROM NN

    print("num_trials: ", num_trials, "|| min_score: ", min_score, "|| min_velocity: ", min_velocity)


    trainingX, trainingY = [], []
    max_reward = -10000
    max_velocity = 10000

    scores = []
    velocities = []

    SHOW_EVERY = 5

    for episode in range(num_trials):
        observation = env.reset()

        trial_reward = 0
        trial_velocity = 0
        game_memory = []
        training_sampleX, training_sampleY = [], []

        for step in range(sim_steps):

            if episode % SHOW_EVERY == 0: env.render()

            if len(observation) == 0:
                action = random.randrange(0, 2)
            else:
                action = model.predict(observation.reshape(1, 3))                               # MODEL PREDICTION

            observation, reward, done, wtff = env.step([action])
            game_memory.append([observation, action])
            trial_reward += reward
            trial_velocity += abs(observation[2])

            training_sampleX.append(observation)
            training_sampleY.append(action)

            if (done):
                print("WON")
                break

        if (trial_reward > max_reward): max_reward = trial_reward

        if (trial_reward > min_score and trial_velocity < min_velocity and trial_reward > max_reward and trial_velocity < max_velocity):        # APPENDING DATA IN TRAINING DATA --> need to fit
            trainingX += training_sampleX
            trainingY += training_sampleY

            scores.append(trial_reward)
            velocities.append(trial_velocity)

        if episode % SHOW_EVERY == 0 and episode != 0: print("episode: ", episode, "saved steps: ", len(trainingY))

            # Change NN

    # print(len(trainingY))
    # print("trainingX[0]): ", trainingX[0])
    # print("trainingY[0]): ", trainingY[0])

    trainingX, trainingY = np.array(trainingX), np.array(trainingY)
    print("Saved steps: ", trainingX.shape)
    print("Best score: ", max_reward)
    print("Average score: {}".format(np.mean(scores)))
    print("Median score: {}".format(np.median(scores)))
    print("Average velocity: {}".format(np.mean(velocities)))
    print("Median velocity: {}".format(np.median(velocities)))

    return trainingX, trainingY





def pendulum_v0():
    start_time = time.time()

    trainingX, trainingY = training_data(10000, -900, 700)  ##########################

    training_time = time.time() - start_time
    training_time_moment = time.time()
    print()
    print("training X: ", trainingX.shape)
    print("training Y: ", trainingY.shape)
    print()
    print()
    print("Getting training data time:", training_time)
    print()
    print()

    model = create_model(1e-3, 0.4)  ##########################
    # model = keras.models.load_model("CartPoleModel_1")
    model_creation_time = time.time() - training_time_moment
    model_creation_time_moment = time.time()
    # print("Model creating time:", model_creation_time)
    # print()

    model.summary()
    print()

    epochs = 10
    history = model.fit(trainingX, trainingY, epochs=epochs)  ##########################
    model_fit_time = time.time() - model_creation_time_moment
    print()
    print("Model fitting time:", model_fit_time)
    print()




    env = gym.make("Pendulum-v0")
    env.reset()

    game_memory = []
    observation = []
    score = 0
    for step in range(199):
        env.render()
        if len(observation) == 0:
            action = random.randrange(0, 2)
        else:
            action = model.predict(observation.reshape(1, 3))
        observation, reward, done, info = env.step([action])
        prev_obs = observation
        game_memory.append([observation, action])
        score += reward
        if done:
            break

    print("Score:", score)


def test_game(trainingX_path, trainingY_path):

    model = create_model_small()

    model.summary()
    print()

    trainingX = np.load(trainingX_path)
    trainingY = np.load(trainingY_path)

    print("trainingX.shape: ", trainingX.shape)

    epochs = 10
    history = model.fit(trainingX, trainingY, epochs=epochs)

    env = gym.make("Pendulum-v0")

    observation = env.reset()

    for step in range(199):
        env.render()

        if len(observation) == 0:
            action = random.randrange(0, 2)
        else:
            action = model.predict(observation.reshape(1, 3))  # MODEL PREDICTION

        observation, reward, done, wtff = env.step([action])

        if (done):
            print("WON")
            break