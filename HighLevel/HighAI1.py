import gym
from Utils import plot_learning_curve
from stable_baselines3.common.monitor import Monitor
from stable_baselines3 import DQN, A2C, PPO
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
from collections import defaultdict
import plottingDennyBritz as plotting
import itertools
import matplotlib
import matplotlib.style
matplotlib.style.use('ggplot')

# from stable_baselines.common.policies import MlpPolicy, MlpLstmPolicy, MlpLnLstmPolicy
# from stable_baselines.common import make_vec_env
# from stable_baselines import ACER

# import torch
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
warnings.filterwarnings("ignore") # get rid of warnings

def eGreedy(Q, e, num):
    # e-greedy based on Q table

    def policy(state):
        actionProb = np.ones(num, dtype=float)*e/num
        best = np.argmax(Q[state]) # get max q value index
        actionProb[best] += 1 - e   # NOT SURE !!!!!
        return actionProb

    return policy

def QLearning(env, episodes, gamma = 0.99, alpha = 0.6, e = 0.1):
    Q = defaultdict(lambda: np.zeros(env.action_space.n))
    # track stats
    stats = plotting.EpisodeStats(episode_lengths = np.zeros(episodes), episode_rewards = np.zeros(episodes))
    policy = eGreedy(Q, e, env.action_space.n)

    for i in range(episodes):
        state = env.reset()
        for t in itertools.count():

            probabilities = policy(state)
            # select action based on policy probabilities
            action = np.random.choice(np.arange(len(probabilities)), p = probabilities)

            # take action and get reward
            newState, reward, done, _ = env.step(action)

            stats.episode_lengths[i] = t
            stats.episode_rewards[i] += reward

            # TD update
            nextBestAction = np.argmax(Q[newState])
            tdTarget = reward + gamma*Q[newState][nextBestAction]
            tdDelta = tdTarget - Q[state][action]
            Q[state][action] += alpha*tdDelta

            if done:
                break
            state = newState
    return Q, stats


if __name__ == "__main__":
    run = "HighAI1"
    modelName = "DQN_1Towers_"
    env = HighEnv1("Validation_1.txt")

    # # Q Learning
    # Q, stats = QLearning(env, 5000)
    # fig1, fig2, fig3 = plotting.plot_episode_stats(stats)
    #
    # fig1.savefig("Fig1.png")
    #
    # fig2.savefig("Fig2.png")
    #
    # fig3.savefig("Fig3.png")

    env = Monitor(env)
    n_actions = env.action_space.n
    policy_kwargs = dict(net_arch=[128, 128], activation_fn=th.nn.modules.activation.LeakyReLU)
    # model = DQN.load(modelName + run + "_best/best_model", env=env)
    # model = DQN("MlpPolicy", env,  target_update_interval=10, verbose=1, policy_kwargs=policy_kwargs, buffer_size=10000, learning_starts=50)
    model = DQN("MlpPolicy", env, learning_rate=0.01, verbose=1, learning_starts=50, policy_kwargs=policy_kwargs, exploration_fraction=0.8)
    eval_callback = EvalCallback(env,  best_model_save_path=modelName + run + "_best",
                                 log_path='./logs/', eval_freq=10,
                                 deterministic=False, render=False)
    model.learn(total_timesteps=1500, log_interval=50, callback=eval_callback)

    model.save(modelName + run)
    print("Saved!\n")

    temp = env.get_episode_rewards()
    n_games = len(temp)
    x = [i+1 for i in range(n_games)]
    figure_file = modelName + run + ".png"
    title = modelName + run
    plot_learning_curve(x, temp, title, figure_file)
    del model  # remove to demonstrate saving and loading


    # now apply to testing
    model = DQN.load(modelName + run + "_best/best_model")
    fileName = "Validation_1.txt"
    # fileName = "MinTowerSim_2020.txt"
    num = 1  # 126 # how many towers are in the file
    env = HighEnv1(fileName=fileName)
    env = Monitor(env)

    obs = env.reset()
    score = 0
    count = 0
    scores = []
    # print(model.policy.net_arch)
    # print(model.policy.action_space.n)
    while True:
        # print(obs)
        action, _states = model.predict(np.reshape(obs, (62,)), deterministic=False)
        obs, rewards, dones, info = env.step(action)
        score += rewards
        if dones:
            obs = env.reset()
            count += 1
            scores.append(score)
            score = 0
        if count >= num:
            x = [i for i in range(len(scores))]
            title = "Validation Pillar Placement AI for " + str(num) + " Towers " + run + "_best policy"
            figure_file = title + ".png"
            plot_learning_curve(x, scores, title, figure_file)
            break
