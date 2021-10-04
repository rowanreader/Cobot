import gym
import numpy as np
from Agents import Agent
from Utils import plot_learning_curve
#from Octomap import Environment


if __name__ == '__main__':
    # env = Environment()
    env = gym.make('Pendulum-v0') # placeholder
    # maxSize, inputShape, dim1, dim2, numJoints, batchSize, alpha, beta, gamma, noise
    agent = Agent(env.observation_space.shape, env, numActions=env.action_space.shape[0]) # just send in shape/size of observation (state, not action)
    numGames = 250
    figureFile = 'plots/pendulum.png'

    bestScore = env.reward_range[0]
    scoreHistory = []
    loadCheckpoint = False

    if loadCheckpoint: # just use old one
        numSteps = 0
        while numSteps < agent.batchSize:
            state = env.reset()
            action = env.action_space.sample() # choose action (random?)
            state_, reward, endFlag, info = env.step(action) # carry out action
            agent.record(state, action, reward, state_, endFlag)
            numSteps += 1
        # now have enough in memory bank to sample/learn, modifies nets
        agent.learn()
        agent.loadModels() # ??? not sure why we're loading models...
        evaluate = True
    else:
        evaluate = False

    for i in range(numGames):
        state = env.reset()
        endFlag = False
        score = 0
        # go until end of episode (either failure or success)
        while not endFlag:
            action = agent.chooseAction(state, evaluate)

            state_, reward, endFlag, info = env.step(action) # observe effects of actions
            score += reward # track total score
            agent.record(state, action, reward, state_, endFlag)
            if not loadCheckpoint:
                agent.learn()
            state = state_
        scoreHistory.append(score)
        aveScore = np.mean(scoreHistory[-100:]) # average of prev 100 entries

        if aveScore > bestScore:
            bestScore = aveScore
            if not loadCheckpoint:
                agent.saveModels()

        print('episode ', i, 'score %.1f' % score, 'avg score %.1f' % aveScore)

    if not loadCheckpoint:
        x = [i + 1 for i in range(numGames)]
        plot_learning_curve(x, scoreHistory, figureFile)


