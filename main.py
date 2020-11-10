

import OpenAi.CartPole.example_test as example_test

import OpenAi.MountainCar.example as MountainCar_example
import OpenAi.MountainCar.sentdex_example as MountainCar_Sendex_example

import OpenAi.Pendulum.testing as Pendulum_testing






import gym







##      ______CART POLE______
#example_test.cart_pole_v1()


##      ______MOUNTAIN CAR______
#mountain_car.test_play_display_info()


# EXAMPLE 1
#print("[+] Playing Random Games")
#df = MountainCar_example.play_random_games(games=1000)


#print("[+] Training NN Model")
#ml_model = MountainCar_example.generate_ml(df)

#print("[+] Playing Games with NN")
#MountainCar_example.play_game(ml_model=ml_model, games=3)


# EXAAMPLE Sentdex
#env = gym.make("MountainCar-v0")
#print("reset discrete_state: ", MountainCar_Sendex_example.get_discrete_state(env.reset()))
#print("env.reset(): ", env.reset())
#print("env.observation_space.low: ", env.observation_space.low)

#MountainCar_Sendex_example.start()


##__________Breakout_________

##Testing
#Pendulum_testing.prep()
#Pendulum_testing.test_games()
#Pendulum_testing.training_data()

#Pendulum_testing.pendulum_v0()
Pendulum_testing.play_and_train()
