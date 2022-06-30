import gym
from gym import spaces
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import SawyerSim
import pickle

# environment for the highest of the high level AIs
# takes in representation of tower, move to complete, picks best place to put pillars
class TopHighEnvironment(gym.Env):
    # NOTE: OBSERVATION_SPACE AND ACTION_SPACE ARE OF TYPE Space (FROM GYM)
    # def __init__(self, observation_space, action_space, reward_range, filled, origins, goal, state=neutralState):
    def __init__(self, fileName):
        self.filename = fileName
        self.file = open(self.fileName, 'rb')

    def step(self, action):
        pass

    def reset(self):
        pass

    def render(self):
        pass