import gym
from Utils import plot_learning_curve
from stable_baselines3 import PPO, SAC, TD3
from stable_baselines3.common.monitor import Monitor
from stable_baselines3 import DDPG
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise

from sb3_contrib import TQC
import torch as th
import pickle
import tensorflow as tf
from Environment import Environment
import numpy as np
import TowerSim as tower
import os
import cProfile, pstats, io
from pstats import SortKey
import warnings

# import torch
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
warnings.filterwarnings("ignore") # get rid of warnings
# Parallel environments
# env = make_vec_env("CartPole-v1", n_envs=4)

# cuda = torch.device('cuda')
def ddpg():
    fileName = "TowerModels.txt"
    f = open(fileName, 'rb')
    temp = pickle.load(f)
    f.close()
    spots = temp[0]
    filled = temp[1]
    origins = temp[2]
    occupied = tower.getOccupied(spots, filled)
    goal = np.float32(np.array([682.75, 402.6875, 141]))
    env = Environment(spots, filled, occupied, origins)
    env = Monitor(env)
    n_actions = env.action_space.shape[-1]
    # action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=np.float32(50) * np.ones(n_actions))
    # model = DDPG("MlpPolicy", env, action_noise=action_noise, verbose=2, learning_rate=0.00001)
    # model.learn(total_timesteps=10000, log_interval=1)
    # model.save("ddpg_sawyer")
    # env = model.get_env()
    # print("Saved")
    # del model # remove to demonstrate saving and loading

    # model = DDPG.load("ddpg_sawyer")

    score = 0
    count = 0
    obs = env.reset()
    while True:
        action, _states = model.predict(obs)
        obs, rewards, dones, info = env.step(action)
        score += rewards
        count += 1
        if dones or count < 1000:
            print(score)
            print(obs)
            if dones:
                print("Success!")
            break

        # env.render()

# if __name__ == '__main__':
#     pr = cProfile.Profile()
#     pr.enable()
#     main()
#     s = io.StringIO()
#     sortby = SortKey.CUMULATIVE
#     ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
#     ps.print_stats()
#     print(s.getvalue())

run = "1000"
prerun = "1"
modelName = "sac"
num = 1000
ver = "direct_med_" + str(num) + "_"
fileName = "TowerModels" + str(num) + ".txt"
f = open(fileName, 'rb')
temp = pickle.load(f)
f.close()
spots = temp[0]
filled = temp[1]
origins = temp[2]
goal = temp[3]
occupied = tower.getOccupied(spots, filled)
env = Environment(spots, filled, occupied, origins, goal, fileName=fileName)
env = Monitor(env)

n_actions = env.action_space.shape[-1]
policy_kwargs = dict(activation_fn=th.nn.Tanh, net_arch=[256, 128])
# policy_kwargs = dict(net_arch=[256, 256, 256])

# model = PPO("MlpPolicy", env, policy_kwargs=policy_kwargs, verbose=1, learning_rate=0.003)
# model = PPO.load("ppo_sawyer_" + ver + prerun, env=env) # train more on pretrained

# model = SAC.load(modelName + "_sawyer_" + ver + run, env=env) # train more on pretrained
# model = SAC("MlpPolicy", env, policy_kwargs=policy_kwargs, learning_rate=0.0003, verbose=1)

# action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))
# model = TD3("MlpPolicy", env, policy_kwargs=policy_kwargs, verbose=1)

# policy_kwargs = dict(n_critics=2, n_quantiles=25)
# model = TQC("MlpPolicy", env, top_quantiles_to_drop_per_net=2, verbose=1, policy_kwargs=policy_kwargs)


# model.learn(total_timesteps=150000, log_interval=num) # for ppo, sac
# model.learn(total_timesteps=100000, log_interval=50)
#
# model.save("ppo_sawyer_" + ver + run)
# model.save(modelName + "_sawyer_" + ver + run)
# print("Saved!\n")

# temp = env.get_episode_rewards()
# n_games = len(temp)
# x = [i+1 for i in range(n_games)]
# figure_file = modelName + ver + run + ".png"
# title = modelName + " " + ver + run
# plot_learning_curve(x, temp, title, figure_file)
# file = open("Scores.txt", 'w')
# file.write(str(temp))
# file.close()
# del model # remove to demonstrate saving and loading

# model = PPO.load(modelName + "_sawyer_" + ver + run)
model = SAC.load(modelName + "_sawyer_" + ver + run)
# model = TQC.load(modelName + "_sawyer_" + ver + run)

# for testing
num = 20
fileName = "TowerModels" + str(num) + ".txt"
 # number of towers in file
f = open(fileName, 'rb')
temp = pickle.load(f)
f.close()
spots = temp[0]
filled = temp[1]
origins = temp[2]
goal = temp[3]
occupied = tower.getOccupied(spots, filled)
env = Environment(spots, filled, occupied, origins, goal, fileName=fileName)
env = Monitor(env)

obs = env.reset()
score = 0
count = 0
scores = []
while True:
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    score += rewards
    if dones:
        obs = env.reset()
        count += 1
        scores.append(score)
        score = 0
    if count >= num:
        x = [i for i in range(len(scores))]
        title = "Test for " + str(num)
        figure_file = title + ".png"
        plot_learning_curve(x, scores, title, figure_file)
        break
        # print(score)
        # print(obs)
    # obs = env.reset()
        # break
    # env.render()