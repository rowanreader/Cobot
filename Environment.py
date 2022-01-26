import gym
from gym import spaces
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import SawyerSim
import pickle
import TowerSim as tower
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
# limit = 5000
# low = np.array([-limit, -limit, -limit]) # x, y, z
# high = np.array([limit, limit, limit])
# shape = np.int
# action_space = spaces.Box(low, high, shape)
# OBSERVATION SPACE VARIES
# observation_space = spaces.Box(0, 2, shape=(1000, 1000), dtype=shape) # octomap is 1000x1000 array of 1s and 0s? goal is 2
neutralState = np.array([1, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]) # joint angles (in radians. i think)

class Environment(gym.Env):
    # NOTE: OBSERVATION_SPACE AND ACTION_SPACE ARE OF TYPE Space (FROM GYM)
    # def __init__(self, observation_space, action_space, reward_range, filled, origins, goal, state=neutralState):
    def __init__(self, spots, filled, occupied, origins, arm=neutralState):
        limit = 1000
        low = np.array([-limit, -limit, -limit])  # x, y, z
        high = np.array([limit, limit, limit])
        # shape = np.int
        self.obsLen = 6  # how big the state is
        self.observation_space = spaces.Box(-limit, limit, shape=(self.obsLen,), dtype=np.int) # this is filled spots, origins, and goal
        self.action_space = spaces.Box(low, high, shape=(3,), dtype=np.int)
        self.reward_range = (-3000, 3000) #reward_range
        self.arm = arm  # state of arm
        self.occupied = occupied  # filled coordinates
        self.filled = filled
        self.origins = origins  # origins of tiles
        self.spots = spots
        self.stepCount = 0
        self.endFlag = 0
        # self.goal = SawyerSim.getGoal(self.spots, self.filled)
        self.goal = np.array([682.75, 402.6875, 141])
        self.outcome = 0 # fail (-1), success (1), error (0)
        self.endPoint, _, _ = SawyerSim.FK(self.arm) # where the end point/hand of the robot is

        self.state = self.minConvert(self.occupied, self.origins, self.goal)



    # levels is 0, 1, or 2
    def getHeight(self, levels):
        height = 45  # pillars in mm
        tileHeight = 2 # thickness of cardboard
        return (levels * height + tileHeight * (levels + 1))


    def minConvert(self,occupied, origins, goal):
        return [goal, self.endPoint]

    # takes in occupied coordinates, origin coordinates, and goal coordinate
    # puts in flattened rectangular array
    def convert(self, occupied, origins, goal):
        final = []
        levels = len(origins)
        floorCount = 0

        for i in range(3): # occupied is array of floors
            level = np.mod(i, levels) # repeat if too few
            floor = occupied[level] # pretend always 3 levels to tower
            countSpots = 0
            for spot in floor:
                # pretend always 3 tiles to each level, 5 spots to each tile = 15

                height = self.getHeight(floorCount)
                newSpot = np.append(spot, height) # get height
                final.append(newSpot) # making 1D array
                countSpots += 1

            # add on to make 5 spots per tile
            for extra in range(15-countSpots):
                final.append(newSpot) # append last/most recent a bunch of times i guess?

            floorCount += 1


        # get origins
        for k in range(3): # 3 floors
            level = np.mod(k, levels)
            floorOrigins = origins[level]
            tiles = len(floorOrigins)
            countFloor = 0
            for j in range(3): #3 tiles
                tileOrigin = floorOrigins[np.mod(j,tiles)]
                height = self.getHeight(countFloor)
                newTile = np.array([tileOrigin[0], tileOrigin[1], height])
                final.append(newTile)
            countFloor += 1

        final.append(goal)

        # print("Final" + str(final))
        return final



    # reward must increase the closer it gets to goal too
    def getReward(self, action):
        thresh = 10 # if within 10 mm close enough
        outcome = self.outcome
        # default -1 for step
        if outcome == -1: # fail
            self.endFlag = True
            print("Failed!")
            alpha = -0.3
            dist = self.getDist(action)
            print("Reward Dist: " + str(dist))
            # print(dist)
            return alpha * dist
        # return -2500 # large negative reward, to make it unattractive to insta-fail
        elif outcome == 0:  # error
            return 0  # must retry, no penalty
        elif outcome == 1:  # success
            # reward based on how close to goal distance is
            alpha = -0.003
            dist = self.getDist(action) # check if absolute distance is close enough
            if dist < thresh:
                self.endFlag = 1
            if self.endFlag == 0:
                # dist = self.getDist(action)
                # print(dist)
                rdist = self.getRelativeDist(action)
                return alpha*rdist  # step reward (small negative)
            print("Succeeded!)")
            return 100  # large positive reward
        print("Shouldn't get here, reward error")
        return 0  # shouldn't get here

    def getDist(self, pt):
        dist = np.linalg.norm(self.goal - pt)
        return dist

    # finds distance between current location and previous location
    # returns difference such that if it is closer now the return value is negative
    def getRelativeDist(self, pt):
        prevDist = self.getDist(self.endPoint)
        dist = self.getDist(pt)
        return dist - prevDist


    # returns state_, reward, endFlag, info
    # state is numpy array, reward is a float64, and endFlag is a bool
    # will have to modify state
    # applies action to self.observation_space to generate new state
    def step(self, action): # carry out action according to state
        while True: # only want to run once, but do need to get goal
            # outcome is 0 (error), -1 (failure), 1 (success)
            # temp should be same as action
            self.outcome, self.arm, temp = SawyerSim.IK(action, self.arm, self.spots, self.filled, self.origins)
            if self.outcome != 0: # should break vast majority of time
                break

        self.stepCount += 1

        info = 0 # placeholder, add debugging info in needed

        reward = self.getReward(temp)

        self.endPoint = temp
        self.state = self.minConvert(self.occupied, self.origins, self.goal)


        # could probably just return self??? but system wants it like this
        # total observation includes both octomap and joint configurations - will need to join better probably
        return self.state, reward, self.endFlag, info


    # gets new state, completely fresh
    # state for our purposes is tower
    def reset(self, readIn, f, fileName):
        self.arm = neutralState # arm is q
        self.stepCount = 0
        self.endFlag = 0
        self.spots = -1
        if readIn:
            # we want to read in the next tower
            try:
                temp = pickle.load(f)
            except EOFError:
                f.close() # close file, reopen, try again
                f = open(fileName, 'rb')
                temp = pickle.load(f)

            self.spots = temp[0]
            self.filled = temp[1]
            self.origins = temp[2]
            self.goal = SawyerSim.getGoal(self.spots, self.filled)
        else:
            f = 0  # placeholder
            while self.spots == -1:
                self.spots, self.filled, self.origins = tower.build()  # all spots, binary array of occupied or not, origins
                if self.spots != -1: # gotta check
                    self.goal = SawyerSim.getGoal(self.spots, self.filled) # get goal, if error, retry
                if self.goal[0] == -1:
                    self.spots = -1

        self.endPoint, _, _ = SawyerSim.FK(self.arm)
        self.occupied = tower.getOccupied(self.spots, self.filled)

        # based on spots, pick one of the unfilled ones from the top floor as a goal
        # FIXED GOAL
        self.goal = np.array([682.75, 402.6875, 141])


        self.state = self.minConvert(self.occupied, self.origins, self.goal) # must be a combo of origins, pillars, and goal

        return f, self.state

    def flatten(self, spots):
        temp = []
        for i in range(3):
            for j in spots[i]:
                # j is now 2D spot, get 3D height and append
                height = self.getHeight(i)
                temp.append(np.append(j,height))
        return np.array(temp)

    # plot goal
    # plot spots
    # draw line in occupied
    # draw arm and endpoint with bubble around it
    # save to file
    def render(self, num, val):
        file = "plots/attempt: " + str(num) + ".jpg"
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        state = self.convert(self.occupied, self.origins, self.goal) # have to reconvert due to minConvert
        occupied = state[0:45]
        spots = self.flatten(self.spots)
        goal = self.goal
        ax.scatter3D(goal[0], goal[1], goal[2], marker='^', color='r')
        ax.scatter3D(spots[:,0], spots[:,1], spots[:,2], marker='.')
        # draw lines for occupied
        for i in occupied:
            ax.plot([i[0], i[0]], [i[1], i[1]], [i[2], i[2] + 45], color='g')

        p, joints, endEffector = SawyerSim.FK(self.arm)
        ax.plot3D(joints[:, 0], joints[:, 1], joints[:, 2], '.-r')
        ax.plot3D(endEffector[:, 0], endEffector[:, 1], endEffector[:, 2], '.-b')
        ax.plot3D(p[0], p[1], p[2], '*c')
        plt.title("Reward: " + str(val))
        fig.savefig(file)
        plt.close(fig)

