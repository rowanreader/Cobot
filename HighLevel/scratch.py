import pickle
from HighTowerSim import Tile
import PlaceTileSim
import numpy as np
# fileName="Validation_25.txt"
fileName="TowerModels1000.txt"
# fileName="TestTileTower.txt"
file = open(fileName, 'rb')

# fileName2 = "TileTowerStable.txt"
# file2 = open(fileName2, 'wb')

# file2 = open("Validation_1.txt", 'wb')
count = 0
total = 0
# level3 = 0
# level2 = 0
while True:
    try:
        tower = pickle.load(file)[0]
        count += 1
        # iterate through all possible spots in the tower, place pillar on each one at a time. if any passes, save
        # length = len(tower[-1][0].spots)
        # for i in range(length):
        #     tower = np.copy(tower)
        #     tower[-1][0].filled[i] = 1
        #     collapsed = PlaceTileSim.simulate(tower)
        #     tower[-1][0].filled[i] = 0  # rather than saving totally diff copy, just change back
        #     if not collapsed:
        #         pickle.dump([tower], file2)
        #         count += 1
        #         print("Saving tower: ", count, " Pillar: ", i)
        #         break  # exit for loop


        # pickle.dump([oldtower], file2)
        # print(count,  oldtower)
        # levels = len(oldtower)
        # if levels == 2:
        #     level2 += 1
        # elif levels == 3:
        #     level3 += 1
        # else:
        #     print("Oops: ", levels)
        # if count == 15:
        #     break
        # count += 1
        # total += 1
    except EOFError:
        break

file.close()
# file2.close()
print(count)
# print(total)
# file2.close()
# print(level2)
# print(level3)
if __name__ == "__main__":
    print("hi")
#
# import gym
# from Utils import plot_learning_curve
# from stable_baselines3 import PPO
# from stable_baselines3.common.env_util import make_vec_env
# from stable_baselines3.common.monitor import Monitor
#
# # Parallel environments
# env = gym.make("CartPole-v1")
# env = Monitor(env)
# model = PPO("MlpPolicy", env, verbose=1, batch_size=32)
# model.learn(total_timesteps=25000)
# temp = env.get_episode_rewards()
# n_games = len(temp)
# x = [i+1 for i in range(n_games)]
# modelName = "ppoCart"
# run = " 1"
# figure_file = modelName + run + ".png"
# title = modelName + run
# plot_learning_curve(x, temp, title, figure_file)
# model.save("ppo_cartpole")
#
# del model # remove to demonstrate saving and loading
#
# model = PPO.load("ppo_cartpole")
#
# obs = env.reset()
# score = 0
# count = 0
# scores = []
# num = 40
# while True:
#     action, _states = model.predict(obs)
#     obs, rewards, dones, info = env.step(action)
#     if dones:
#         obs = env.reset()
#         count += 1
#         scores.append(score)
#         score = 0
#
#     if count >= num:
#         x = [i for i in range(len(scores))]
#         title = "Validation Pillar Placement AI for " + str(num) + " Towers " + run + "_best policy"
#         figure_file = title + ".png"
#         plot_learning_curve(x, scores, title, figure_file)
#         break

# for i in range(3):
#     print("Hi")
#     quit()