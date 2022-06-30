import sys

import gym
from gym import spaces
import numpy as np
import pickle
import random
import CoppeliaTower
from HighTowerSim import Tile, getClusters, getTile, getRotation, tupleTransform, getHeight, getRotateMat

class HighEnv1(gym.Env):
    def __init__(self, fileName="MinTowerSim_100.txt"):
        self.fileName = fileName
        self.file = open(self.fileName, 'rb')
        # used to build tower

        self.towerCount = 0
        self.oldtower = None  # placeholder, immediately changed

        self.getNext()  # assigns tower
        # used as AI input
        self.modifiedtower = self.copy(self.oldtower)
        # not needed for this - just go to end of move
        self.bag = 20  # keep track of how many pieces are available
        self.myPillars = [0, 1, 2, 0, 1] #self.getPillars(5)
        self.campPillars = self.getPillars(6)
        self.partnerPillars = self.getPillars(5)
        self.card = 3 # random.randrange(1, 4)  # random, 1, 2, or 3 = # of pillars to place
        self.count = 0

        self.obsLen = 62 # change to 57, remove pillars
        self.state = self.getState()
        # self.state = 0
        self.history = []  # keep track of pillars and tiles placed this turn and their order

        # state is tower rep and pillars
        # tower is upto 3 levels of up to 5 spots
        # tower = 15 * 3 (coords flattened + filled bool) + 3(id) + 3*3 (origin of floors (exclude z)= 57 (with padding)
        # pillars = 5 (mine) + 11 (partners + camp) = 16
        # total = 73
        limit = 800  # everything is in mm
        # self.low = np.array([0, 0, 0, 0, 0])  # assume softmax, so between 0 and 1
        # self.high = np.array([1, 1, 1, 1, 1])
        self.observation_space = spaces.Box(-limit, limit, shape=(self.obsLen,))
        self.action_space = spaces.Discrete(5)  # numbers 0-5, where 5 represents giving up (no move possible)
        self.reward_range = (-80, 10)

    # make a completely new copy of the given tower
    def copy(self, tower):
        newTower = []
        try:
            for level in tower:
                floor = []
                for tile in level:
                    newTile = Tile(tile.id, tile.spots, rotation=tile.rotation,
                                   origin=tile.origin, outline=tile.outline, colours=tile.colours)
                    newTile.worldSpots = np.copy(tile.worldSpots)
                    newTile.filled = np.copy(tile.filled)
                    floor.append(newTile)
                newTower.append(floor)
            return newTower
        except Exception as e:
            print(self.towerCount)
            print("HI")

    def getNext(self):
        try:
            self.towerCount += 1
            self.oldtower = pickle.load(self.file)[0]

        except EOFError:
            self.file.close()  # close file, reopen, try again
            self.file = open(self.fileName, 'rb')
            self.oldtower = pickle.load(self.file)[0]

    # generate random array of num length, with numbers 1-5 according to probability dist
    # 0 = yellow (freq = 39%)
    # 1 = red (freq = 26%)
    # 2 = black (freq = 16%)
    # 3 = white (freq = 11%)
    # 4 = blue (freq = 8%)
    def getPillars(self, num):
        choices = [0, 1, 2, 3, 4]
        freq = [0.39, 0.26, 0.16, 0.11, 0.08]
        return random.choices(choices, freq, k=num)

    def getState(self):
        index = 0
        ids = ["Tile01", "Tile02", "Tile03", "Tile04", "Tile05", "Tile06", "Tile07", "Tile08", "Tile09", "Tile10",
               "Tile11", "Tile12", "Tile13", "Tile14", "Tile15", "Tile16", "Tile17", "Tile18"]
        state = np.zeros(self.obsLen)
        floorCount = 0
        try:
            for floor in self.modifiedtower:  # should only have 1 item in each
                for tile in floor:
                    id = ids.index(tile.id)
                    state[index] = id  # keep within range of others
                    index += 1
                    state[index:index+3] = tile.origin
                    index += 3
                    count = 0  # make sure there are always 5 spots
                    for spot in tile.spots:  # use local spots

                        state[index:index+2] = spot # local
                        if tile.filled[count]:
                            state[index+2] = 1  # filled, otherwise leave as 0
                        index += 3
                        count += 1


                    for i in range(count, 5):
                        index += 3  # leave as 0s
                floorCount += 1

            for _ in range(floorCount, 3):
                index += 19

            # now add pillars
            state[index:index+5] = self.myPillars
            index += 5
            # state[index:index+5] = self.partnerPillars
            # index += 5
            # state[index:index+6] = self.campPillars
        except Exception as e:
            print(e)
        return state



    def getCoord(self, action):
        # if action is 5, agent is chosing to pass and restart = large neg reward
        if action == 5:
            return None

        # action is 0-4 representing location of pillars on top floor (order is set)
        tile = self.modifiedtower[-1][0]
        # check that this tile has that many actions. if not, return -1
        if len(tile.spots) <= action:
            return None
        # rotate = getRotateMat(tile.rotation)
        # coord = tupleTransform(tile.spots[action], tile.origin[0:2], rotate)  # includes z
        coord = np.copy(tile.spots[action])  # avoid alliasing
        filled = tile.filled[action]
        colour = tile.colours[action]
        # ignore colour - no choice for this agent
        # if colour not in self.myPillars:
        #     return None  # don't have that colour pillar, try again
        # else:
        #     self.myPillars.remove(colour)  # removes 1 instance of that number/colour
        #     newCol = self.getPillars(1)
        #     self.myPillars.append(newCol[0])  # adds 1 pillar
        if filled:
            return None  # spot is filled, illegal action, state won't change, try again
        return coord

    def getReward(self, collapse):
        if collapse:
            print("Failed!")
            return -10
        else:
            # print("Success!")
            return 5

    # first checks if all spots on the top layer are filled
    # if so, check if numLevels < 3
    # if numLevels = 3, just return endflag = true
    # otherwise add tile to sim. Try 5 times with noise
    def addFloor(self):
        tile = self.modifiedtower[-1][0]
        filled = tile.filled
        if 0 in filled:
            return False, 0  # still spots left, no additional floor

        numLevels = len(self.modifiedtower)
        if numLevels == 3:
            print("Success!!!!!!")
            return True, 10  #50 # end episode without adding next tile


        # if we get here, we need to add another tile - use clusering, add noise, apply hull overlap, add noise
        # try 5 times
        for _ in range(5):
            center = getClusters(tile.worldSpots, tile.filled, 1)  # center is just x and y
            # pick random tile, 1-18 -> 0-17 inclusive
            tileIndex = random.sample(range(0, 18), 1)[0]
            newTile = getTile(tileIndex)
            # set tile properties - add noise to x and y, make sure it's 3D
            newTile.origin = np.append(np.add(center, [random.randint(-3, 4), random.randint(-3, 4)]), getHeight(numLevels))  # add noise, up to 3 mm in any dir
            newTile.rotation = getRotation(newTile, [newTile.origin], tile.worldSpots)
            rotate = getRotateMat(newTile.rotation)
            for spot in newTile.spots:
                tempSpot = tupleTransform(spot, newTile.origin, rotate)
                newTile.worldSpots.append(np.append(tempSpot, getHeight(numLevels)))  # include height in the worldSpots array
            newTile.filled = np.zeros(len(newTile.spots)) # empty to start
            tempHistory = self.history + [[1, newTile]]
            # simulate old tower with addons
            collapse = CoppeliaTower.simulate(self.oldtower, tempHistory)
            if not collapse:
                self.history = tempHistory
                np.append(self.modifiedtower, [newTile])
                return False, -5  # 15  # added, keep going (additional reward for risky move)
            # if made it here, didn't work, reset worldSpots? Should have been taken care of already but...
            newTile.worldSpots = []
        return True, -4  # 30 # didn't work, end episode, consider a success?


    def step(self, action):
        # FOR RANDOM AGENT ONLY
        # action = random.randint(0, 5)
        # collapse = False
        location = self.getCoord(action)  # get coordinate of chosen spot - must be local
        self.count += 1
        endflag = 0
        # modify tower to reflect filled (must do after simulation so that simulated in order)
        if action == 5:  # choose to pass, large negative reward

            reward = -60
            endflag = 1
            print("Chose to pass")

        elif location is None:  # not valid move
            print("Illegal!")
            reward = -15  # negative reward, no change in state, return
        else:  # only modify tower if mover is valid/successful

            tile = self.modifiedtower[-1][0]
            self.history.append([0, location, tile.id])  # 0 means it's a pillar object

            # instead of simulating, just use hardcode
            # build tower so that it stands, then add pillar. Observe collapse
            collapse = CoppeliaTower.simulate(self.oldtower, self.history)
            # if action == 0 and self.card == 3:
            #     self.state = 1
            #     collapse = True
            # elif action == 3 and self.card == 3:
            #     self.state = 2
            #     collapse = False
            # elif action == 0 and self.card == 2:
            #     self.state = 3
            #     collapse = False


            if collapse == -1:
                print("Error in sim!")
                quit()

            self.modifiedtower[-1][0].filled[action] = 1
            endflag, tempReward = self.addFloor()  # if finished 3rd level, success (endFlag = 1) else add next tile
            # temp reward gives reward for cases where ending without card == 0

            self.card -= 1

            reward = self.getReward(collapse)
            reward += tempReward  # from adding floor
            if collapse or self.card == 0:
                endflag = 1  # finished episode

            if self.card == 0 and not collapse:
                # reward += 60
                reward += 30
                print("Success!")

        info = dict()
        self.state = self.getState()

        if endflag:
            self.count = 0
        # print(action)
        if self.count > 20 and endflag == 0:
            reward = 0
            endflag = 1  # give up
            print("Give up!!")

        print("Action:", action)
        print("Reward:", reward)
        # print("State:", self.state)
        print()
        return self.state, reward, endflag, info


    def reset(self):
        self.getNext()  # assigns tower

        self.modifiedtower = self.copy(self.oldtower)
        # new set of pillars
        self.myPillars = [0, 1, 2, 0, 1] #self.getPillars(5)
        # self.campPillars = self.getPillars(6)
        # self.partnerPillars = self.getPillars(5)
        self.card = 3  #random.randrange(1, 4)  # random, 1, 2, or 3 = # of pillars to place
        self.count = 0
        self.history = []


        # self.state = 0
        self.state = self.getState()
        return np.float32(self.state)

