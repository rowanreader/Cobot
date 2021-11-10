import gym
from gym import spaces
import numpy as np
import SawyerSim
# for low level AI high should be max
# class Space():
#     def __init__(self, high, low, shape):
#         self.high = high
#         self.low = low
#         self.shape = shape  # number of actions (3 = x, y, z)
#
#     # returns single action
#     def sample(self): # returns sample of actions
#         pass
#
#
# class observation_space():
#     def __init__(self, shape):
#         self.shape = shape # input shape of state
limit = 500
low = np.array([-limit, -limit, -limit]) # x, y, z
high = np.array([limit, limit, limit])
shape = np.int
action_space = spaces.Box(low, high, shape)
observation_space = spaces.Box(0, 2, shape=(1000, 1000), dtype=shape) # octomap is 1000x1000 array of 1s and 0s? goal is 2
neutralState = np.array([1, 1, 1, 1, 1, 1, 1]) # joint angles (in radians. i think)

class Environment(gym.Env):
    # NOTE: OBSERVATION_SPACE AND ACTION_SPACE ARE OF TYPE Space (FROM GYM)
    def __init__(self, observation_space, action_space, reward_range, state=neutralState):
        self.observation_space = observation_space # this is an octomap
        self.action_space = action_space
        self.reward_range = reward_range
        self.state = state # state of arm
        self.stepCount = 0
        self.endFlag = 0


    def getReward(self):
        # default -1 for step
        # if collision, large negative reward, set endFlag to True
        return 0

    # returns state_, reward, endFlag, info
    # state is numpy array, reward is a float64, and endFlag is a bool
    # will have to modify state
    # applies action to self.observation_space to generate new state
    def step(self, action): # carry out action according to state
        self.state = SawyerSim.IK(action, self.state)
        self.stepCount += 1
        # NEED TO MODIFY MAP SOMEHOW

        self.endFlag = 0 # NEED TO CHECK IF GOAL REACHED
        info = 0 # placeholder, add debugging info in needed

        reward = self.getReward()

        # could probably just return self??? but system wants it like this
        # total observation includes both octomap and joint configurations - will need to join better probably
        return [self.observation_space, self.state], reward, self.endFlag, info

    # gets new state, completely fresh
    # state for our purposes is octomap (goal location coloured differently?)
    def reset(self):
        self.state = neutralState # SHOULD PROBABLY MAKE RANDOM
        self.stepCount = 0
        self.endFlag = 0

    # display map and end position
    def render(selfself, mode='h', close = False):
        pass