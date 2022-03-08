import gym
import numpy as np

import SawyerSim
from Utils import plot_learning_curve
from PendAgents import Agent
import warnings

warnings.filterwarnings("ignore") # get rid of warnings

printMe = False
readIn = True

if __name__ == '__main__':
    numGames = 200
    # transform = [400, 400, 100] # translate all points in the tower by this much


    env = gym.make('Pendulum-v0') # placeholder

    agent = Agent(env.observation_space.shape, env, numActions=env.action_space.shape[0]) # just send in shape/size of observation (state, not action)

    figureFile = 'plots/pendLowLevel.png'
    lossFile = 'plots/pendLossLowLevel.png'

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
            action = agent.chooseAction(state, evaluate)
            # print("1 action: " + str(action))
            if printMe:
                print(action)
            state_, reward, endFlag, info = env.step(action) # observe effects of actions
            score += reward # track total score
            agent.record(state, action, reward, state_, endFlag)
            if not loadCheckpoint:
                agent.learn()
            state = state_
            count += 1

        scoreHistory.append(score)
        aveScore = np.mean(scoreHistory[-100:]) # average of prev 100 entries
        if aveScore > bestScore:
            bestScore = aveScore
            if not loadCheckpoint:
                agent.saveModels()

        env.score = score
        # env.render()
        print(state)
        print('episode ', i, 'score %.1f' % score, 'avg score %.1f' % aveScore) # always print
        print(" ")

    if printMe:
        print(env.goal, i)
    if not loadCheckpoint:
        x = [i + 1 for i in range(numGames)]
        title = "Score"
        plot_learning_curve(x, scoreHistory, title, figureFile)


    env.close()


