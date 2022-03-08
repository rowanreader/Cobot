# keeps track of replay buffer

import numpy as np

class ReplayBuffer:
    def __init__(self, maxSize, inputShape, numActions=1):
        self.memSize = maxSize
        self.counter = 0 # initialize couter to track index

        # initialize actual storage stuff
        # need to track initial state, action, reward, and resulting state
        self.initialState = np.zeros((self.memSize, *inputShape)) # the * does the equivalent of ' '.join(map(str,inputShape))
        self.action = np.zeros((self.memSize, numActions))
        self.reward = np.zeros(self.memSize)
        self.endState = np.zeros((self.memSize, *inputShape))
        # store whether or not this new state is end of episode (collision or success)
        self.terminal = np.zeros(self.memSize, dtype=np.bool)


    # actually store the transition
    # initial state, action taken, reward receieved, new state, and whether it was the end of the episode
    def store(self, state, action, reward, state_, endFlag):

        index = self.counter % self.memSize
        self.initialState[index] = state
        self.action[index] = action
        self.reward[index] = reward
        self.endState[index] = state_
        self.terminal[index] = endFlag

        self.counter += 1

    # sample from memory buffer
    # returns states, actions, rewards, new states, and terminal flags
    def sample(self, numMems):
        # if the counter is larger than memSize then the limit is the memSize. Otherwise the limit is the counter
        limit = min(self.counter, self.memSize)
        # sample from 0-limit numMems times with no repeats
        sampleIndexes = np.random.choice(limit, numMems, replace=False)

        states = self.initialState[sampleIndexes]
        actions = self.action[sampleIndexes]
        rewards = self.reward[sampleIndexes]
        states_ = self.endState[sampleIndexes]
        flags = self.terminal[sampleIndexes]

        return states, actions, rewards, states_, flags







