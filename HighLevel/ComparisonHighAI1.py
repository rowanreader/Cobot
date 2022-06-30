import gym
from Utils import plot_learning_curve
from stable_baselines3.common.monitor import Monitor
from stable_baselines3 import DQN
from stable_baselines3.common.noise import NormalActionNoise
import torch as th
import pickle
from HighEnvs import HighEnv1
import numpy as np
import HighTowerSim as tower
from HighTowerSim import Tile
import os
import warnings
from stable_baselines3.common.callbacks import EvalCallback

fileName = "Validation_25.txt"
num = 26  # how many towers are in the file
env = HighEnv1(fileName=fileName)
env = Monitor(env)

obs = env.reset()
score = 0
count = 0
scores = []
run = "HighAI1"

if __name__ == "__main__":

    policy_kwargs = dict(net_arch=[128, 128])
    model = DQN("MlpPolicy", env, learning_rate=0.0003, verbose=1, policy_kwargs=policy_kwargs)
    while True:
        # change env to give random action
        action, _states = model.predict(obs, deterministic=False)
        obs, rewards, dones, info = env.step(action)
        score += rewards
        if dones:
            obs = env.reset()
            count += 1
            scores.append(score)
            score = 0
        if count >= num:
            x = [i for i in range(len(scores))]
            # title = "Validation Random Agent for " + str(num) + " Towers " + run + "_best policy"
            title = "Validation Random Agent for " + str(num) + " Towers"
            figure_file = title + ".png"
            plot_learning_curve(x, scores, title, figure_file)
            break
