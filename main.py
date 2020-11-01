

import OpenAi.CartPole.example_test

import OpenAi.MountainCar.example as MountainCar_example









##      ______CART POLE______

#example_test.cart_pole_v1()


##      ______MOUNTAIN CAR______

#mountain_car.test_play_display_info()

print("[+] Playing Random Games")
df = MountainCar_example.play_random_games(games=1000)

print("[+] Training NN Model")
ml_model = MountainCar_example.generate_ml(df)

print("[+] Playing Games with NN")
MountainCar_example.play_game(ml_model=ml_model, games=3)