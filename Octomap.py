from gym import spaces
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
low = 0
high = 0
shape = 0
action_space = spaces.Box(low, high, shape)
observation_space = spaces.Box(low, high, shape)

class Environment():
    # NOTE: OBSERVATION_SPACE AND ACTION_SPACE ARE OF TYPE Space (FROM GYM)
    def __init__(self, observation_space, action_space, reward_range, state):
        self.observation_space = observation_space
        self.action_space = action_space
        self.reward_range = reward_range
        self.state = state

    # returns state_, reward, endFlag, info
    # state is numpy array, reward is a float64, and endFlag is a bool
    # will have to modify state
    def step(self, action): # carry out action according to state
        pass

    # gets new state, completely fresh
    # state for our purposes is octomap (goal location coloured differently?)
    def reset(self):
        pass