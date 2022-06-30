import pickle
from HighTowerSim import Tile

# fileName="Validation_25.txt"
# # fileName="MinTowerSim_100.txt"
# file = open(fileName, 'rb')
# file2 = open("Validation_1.txt", 'wb')
# count = 0
# # level3 = 0
# # level2 = 0
# for i in range(0, 1):
#     try:
#         oldtower = pickle.load(file)[0]
#         pickle.dump([oldtower], file2)
#         # print(count,  oldtower)
#         # levels = len(oldtower)
#         # if levels == 2:
#         #     level2 += 1
#         # elif levels == 3:
#         #     level3 += 1
#         # else:
#         #     print("Oops: ", levels)
#         if count == 25:
#             break
#         count += 1
#
#     except EOFError:
#         break
#
# file.close()
# file2.close()
# print(level2)
# print(level3)

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