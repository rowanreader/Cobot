import gym
import numpy as np

import SawyerSim
from Agents import Agent
from Utils import plot_learning_curve
from Environment import Environment
import TowerSim as tower
import pickle
import warnings
import os
from numba import jit
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
warnings.filterwarnings("ignore") # get rid of warnings

printMe = False
readIn = True

if __name__ == '__main__':
    numGames = 200
    # transform = [400, 400, 100] # translate all points in the tower by this much

    fileName = "TowerModels.txt"
    goal = [-1]
    while goal[0] == -1:

        if readIn:
            f = open(fileName, 'rb')
            temp = pickle.load(f)
            spots = temp[0]
            filled = temp[1]
            origins = temp[2]
        else:
            f = 0  # placeholder
            spots, filled, origins = tower.build()
        goal = SawyerSim.getGoal(spots, filled)
        occupied = tower.getOccupied(spots, filled)

    goal = np.array([682.75, 402.6875, 141])
    env = Environment(spots, filled, occupied, origins)
    env.file = f
    # env = gym.makeobservation('Pendulum-v1') # placeholder

    agent = Agent(env.observation_space.shape, env, numActions=env.action_space.shape[0]) # just send in shape/size of observation (state, not action)

    figureFile = 'plots/LowLevel.png'
    lossFile = 'plots/LossLowLevel.png'

    bestScore = env.reward_range[0]
    scoreHistory = []
    loadCheckpoint = False

    if loadCheckpoint: # just use old one
        numSteps = 0
        while numSteps < agent.batchSize:
            state = env.reset() # state is [self.occupied, self.origins, self.goal]
            action = env.action_space.sample() # choose action (random?)
            state_, reward, endFlag, info = env.step(action) # carry out action
            if printMe:
                print(action)
            agent.record(state, action, reward, state_, endFlag, env.obsLen)
            numSteps += 1
        # now have enough in memory bank to sample/learn, modifies nets
        agent.learn()
        agent.loadModels() # ??? not sure why we're loading models...
        evaluate = True
    else:
        evaluate = False

    thresh = 300
    for i in range(numGames):
        state = env.reset()
        endFlag = False
        score = 0
        count = 0
        # go until end of episode (either failure or success)
        while not endFlag and count < thresh:
            action = agent.chooseAction(state, env.obsLen, evaluate)
            # print("1 action: " + str(action))
            if printMe:
                print(action)
            state_, reward, endFlag, info = env.step(action) # observe effects of actions
            score += reward # track total score
            agent.record(state, action, reward, state_, endFlag, env.obsLen)
            if not loadCheckpoint:
                agent.learn()
            state = state_
            count += 1
        if count == thresh:
            print("Took too long")
            dist = env.getDist(action)
            print("Distance: " + str(dist))
            # rdist = env.getRelativeDist(action)
            score += dist*-0.5 # essentially failed
        scoreHistory.append(score)
        aveScore = np.mean(scoreHistory[-100:]) # average of prev 100 entries
        if aveScore > bestScore:
            bestScore = aveScore
            if not loadCheckpoint:
                agent.saveModels()
        dist = env.getDist(action)
        # print("Goal: " + str(goal))
        print("Final Action: " + str(action))
        # if np.isnan(score):
        #     print("Oops")
        env.score = score
        env.render()
        print(state)
        print('episode ', i, 'score %.1f' % score, 'avg score %.1f' % aveScore) # always print
        print(" ")

    if printMe:
        print(env.goal, i)
    if not loadCheckpoint:
        x = [i + 1 for i in range(numGames)]
        title = "Score"
        plot_learning_curve(x, scoreHistory, title, figureFile)
        x = [i+1 for i in range(len(agent.actor.lossRec))]
        title = "Loss"
        plot_learning_curve(x, agent.actor.lossRec, title, lossFile)

    env.close()


