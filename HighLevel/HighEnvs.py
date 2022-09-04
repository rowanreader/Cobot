import sys

import gym
from gym import spaces
import numpy as np
import pickle
import random
import CoppeliaTower
from HighTowerSim import Tile, getClusters, getTile, getRotation, tupleTransform, getHeight, getRotateMat
from stable_baselines3 import SAC
# from DQNHigh1 import Autoencoder
from tensorflow.keras.models import Model
from tensorflow.keras import layers, losses
import tensorflow as tf
from stable_baselines3.common.monitor import Monitor
from placeTileEnv import TileEnv
import warnings
warnings.filterwarnings("ignore")
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


obsLen = 38



class Autoencoder(Model):
  def __init__(self, latent_dim):
    super(Autoencoder, self).__init__()
    self.latent_dim = latent_dim
    self.encoder = tf.keras.Sequential([
      layers.Dense(32, activation='relu'),
      layers.Dense(32, activation='relu'),
      layers.Dense(latent_dim, activation='relu'),
    ])

    # re-expand to 35 values
    self.decoder = tf.keras.Sequential([
      layers.Dense(32, activation='relu'),
      layers.Dense(32, activation='relu'),
      layers.Dense(obsLen, activation='linear')
    ])

  def call(self, x):
    encoded = self.encoder(x)
    decoded = self.decoder(encoded)
    return decoded


class HighEnv1(gym.Env):
    def __init__(self, fileName="MinTowerSim_1000.txt"):
        self.latent_dim = 38 # 10  # for autoencoder
        # self.autoencoder = Autoencoder(self.latent_dim)
        # self.autoencoder.load_weights("Autoencoder")
        # self.autoencoder.compile(optimizer='adam', loss=losses.MeanSquaredError(), metrics=['accuracy'])

        self.model = SAC.load("SAC1000_TilePlace_best/best_model")  # tile placement ai
        self.totalCount = 1  # how often to retrain autoencoder
        self.fileName = fileName
        self.file = open(self.fileName, 'rb')
        # used to build tower
        self.maxTowers = 300  # how many towers to train on (too many in file)
        self.towerCount = 0
        self.oldtower = None  # placeholder, immediately changed
        self.illegalMoves = 0
        self.recordIllegal = []
        self.getNext()  # assigns tower
        # used as AI input
        self.modifiedtower = self.copy(self.oldtower)
        # not needed for this - just go to end of move
        # self.myPillars = [1, 1, 2, 2, 3]
        self.myPillars = self.getPillars(5)  #
        while self.checkEnd():
            self.myPillars = self.getPillars(5)
            print("Re-drawing Pillars")
        self.campPillars = self.getPillars(6)
        self.partnerPillars = self.getPillars(5)
        # self.card = 6  # random.randrange(1, 4)  # random, 1, 2, or 3 = # of pillars to place
        self.count = 0

        self.obsLen = self.latent_dim #obsLen  # change to 57, remove pillars
        self.state = self.getState()
        # self.state = 0
        self.history = []  # keep track of pillars and tiles placed this turn and their order
        self.recordTowers = []  # keep a record of towers to retrain autoencoder on
        # state is tower rep and pillars
        # tower is upto 3 levels of up to 5 spots
        # tower = 15 * 3 (coords flattened + filled bool) + 3(id) + 3*3 (origin of floors (exclude z)= 57 (with padding)
        # pillars = 5 (mine) + 11 (partners + camp) = 16
        # total = 73
        limit = 2  # everything is in mm
        # self.low = np.array([0, 0, 0, 0, 0])  # assume softmax, so between 0 and 1
        # self.high = np.array([1, 1, 1, 1, 1])
        self.observation_space = spaces.Box(-1, 6, shape=(self.obsLen,))
        self.action_space = spaces.Discrete(5)  # numbers 0-5, where 5 represents giving up (no move possible)
        self.reward_range = (-16, 16)

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
            if self.towerCount >= self.maxTowers:
                self.file.close()
                self.file = open(self.fileName, 'rb')
                self.towerCount = 0
            self.towerCount += 1
            self.totalCount += 1
            self.oldtower = pickle.load(self.file)[0]

            # if self.totalCount % 1000 == 0:
            #     # retrain auto encoder
            #     print("...............Training Autoencoder...................")
            #     self.autoencoder.fit(self.recordTowers, self.recordTowers, epochs=500, shuffle=True)
            #     self.autoencoder.save_weights("Autoencoder2")
            #     self.recordTowers = []
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
        choices = [1, 2, 3, 4, 5]
        freq = [0.39, 0.26, 0.16, 0.11, 0.08]
        pillars = random.choices(choices, freq, k=num)
        pillars.sort()
        return pillars

    # returns full 35 value state
    def getFullState(self):
        index = 0
        ids = ["Tile01", "Tile02", "Tile03", "Tile04", "Tile05", "Tile06", "Tile07", "Tile08", "Tile09", "Tile10",
               "Tile11", "Tile12", "Tile13", "Tile14", "Tile15", "Tile16", "Tile17", "Tile18"]
        state = np.zeros(obsLen)
        floorCount = 0
        try:
            for floor in self.modifiedtower:  # should only have 1 item in each
                for tile in floor:
                    id = ids.index(tile.id)
                    state[index] = 0 #id  # keep within range of others
                    index += 1
                    state[index:index + 3] = [0, 0, 0] # [round(i / 500, 2) for i in tile.origin]
                    index += 3
                    state[index] = 0 #tile.rotation
                    index += 1
                    # state[index] = np.cos(tile.rotation)
                    # index += 1
                    count = 0  # make sure there are always 5 spots
                    for spot in tile.spots:  # use local spots

                        # state[index:index+2] = spot # local
                        if tile.filled[count]:
                            # state[index+2] = 1  # filled, otherwise leave as 0
                            state[index] = -1  # filled, otherwise leave as 0
                        else:
                            state[index] = tile.colours[count]
                        index += 1
                        count += 1

                    for i in range(count, 5):
                        state[index] = 6  # ghost spots are 6
                        index += 1
                floorCount += 1

            for _ in range(floorCount, 3):
                index += 10
                # index += 5
            # now add pillars
            state[index:index + 5] = self.myPillars  # [x+1 for x in self.myPillars]
            index += 5
            # state[index:index+5] = self.partnerPillars
            # index += 5
            # state[index:index+6] = self.campPillars
        except Exception as e:
            print(e)

        return list(state)


    def getState(self):
        index = 0
        ids = ["Tile01", "Tile02", "Tile03", "Tile04", "Tile05", "Tile06", "Tile07", "Tile08", "Tile09", "Tile10",
               "Tile11", "Tile12", "Tile13", "Tile14", "Tile15", "Tile16", "Tile17", "Tile18"]
        state = np.zeros(obsLen)
        # state = np.zeros(10)
        floorCount = 0
        prevRot = 0
        prevOri = [0, 0, 0]
        try:
            for floor in self.modifiedtower:  # should only have 1 item in each
                for tile in floor:
                    id = ids.index(tile.id)
                    state[index] = id  # keep within range of others
                    index += 1
                    tempOri = [x-y for x, y in zip(tile.origin, prevOri)]
                    state[index:index+3] = [round(i/500, 2) for i in tempOri]
                    index += 3
                    temp = np.deg2rad(tile.rotation)
                    state[index] = np.sin(temp)
                    index += 1
                    state[index] = np.cos(temp)
                    index += 1

                    prevRot = tile.rotation
                    prevOri = tile.origin
                    # state[index] = np.cos(tile.rotation)
                    # index += 1
                    count = 0  # make sure there are always 5 spots
                    for spot in tile.spots:  # use local spots

                        # state[index:index+2] = spot # local
                        if tile.filled[count]:
                            # state[index+2] = 1  # filled, otherwise leave as 0
                            state[index] = -1  # filled, otherwise leave as 0
                        else:
                            state[index] = tile.colours[count]
                        index += 1
                        count += 1


                    for i in range(count, 5):
                        state[index] = 6  # ghost spots are 6
                        index += 1
                floorCount += 1

            for _ in range(floorCount, 3):
                index += 11
                # index += 5
        # try:
        #     tile = self.modifiedtower[-1][0]
        #     count = 0
        #     for spot in tile.spots:
        #         if tile.filled[count]:
        #             # state[index+2] = 1  # filled, otherwise leave as 0
        #             state[index] = -1  # filled, otherwise leave as 0
        #         else:
        #             state[index] = tile.colours[count]
        #         index += 1
        #         count += 1
        #     for i in range(count, 5):
        #         state[index] = 6  # ghost spots are 6
        #         index += 1

            # now add pillars
            state[index:index+5] = self.myPillars # [x+1 for x in self.myPillars]
            index += 5

            # state[index:index+5] = self.partnerPillars
            # index += 5
            # state[index:index+6] = self.campPillars
        except Exception as e:
            print(e)

        # encodedState = self.autoencoder.encoder(np.array([state])).numpy()
        # return encodedState[0] #[x/6 for x in state]
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
        if filled:
            return None  # spot is filled, illegal action, state won't change, try again

        # ignore colour - no choice for this agent
        if colour not in self.myPillars:
            # print("Hi")
            return None  # don't have that colour pillar, try again
        else:
            # get index of colour
            for i in range(5):
                if self.myPillars[i] == colour:
                    self.myPillars[i] = self.getPillars(1)[0]  # replace w new colour
                    break  # exit loop

        return coord

    def getReward(self, collapse):
        if collapse:
            print("Failed!")
            return -10
        else:
            # print("Success!")
            return 15

    # just gets x and  y coords of pillars and adds id of new tile
    def getTileState(self, id):
        tile = self.modifiedtower[-1][0]
        state = np.zeros(11)
        index = 0
        for spot in tile.spots:
            state[index:index + 2] = spot
            index += 2
        state[10] = id
        state = [x / 50 for x in state]
        return state


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
            # print("Finished 3 levels")
            return True, 10  #50 # end episode without adding next tile


        # select new tile, try 3 times w diff tiles
        for a in range(3):
            tileIndex = random.randint(0, 17)
            newTile = getTile(tileIndex)

            # env2 = TileEnv("TileTower.txt", 0)
            # env2 = Monitor(env2)
            # _ = env2.reset()
            obs2 = self.getTileState(tileIndex)
            action2, _states2 = self.model.predict(obs2, deterministic=False)
            # obs2, rewards2, dones2, info2 = env2.step(action2)
            newTile.worldSpots = []
            # add to tower
            newTile.origin = [action2[0], action2[1], getHeight(numLevels)]
            newTile.rotation = action2[2]
            rotate = getRotateMat(newTile.rotation)
            for spot in newTile.spots:
                tempSpot = tupleTransform(spot, newTile.origin, rotate)
                newTile.worldSpots.append(np.append(tempSpot, getHeight(numLevels)))
                tempHistory = self.history + [[1, newTile]]
                # simulate old tower with addons

            collapse = CoppeliaTower.simulate(self.oldtower, tempHistory)
            # collapse = 0  # didn't collapse
            if not collapse:
                self.history = tempHistory
                self.modifiedtower.append([newTile])
                return False, 15  # 15  # added, keep going (additional reward for risky move)

        # otherwise give up
        return True, 5  # end episode, give minimal reward


        # addFloorTiles = [2, 3, 4, 9, 11, 12]
        # # if we get here, we need to add another tile - use clusering, add noise, apply hull overlap, add noise
        # # try 5 times
        # for _ in range(5):
        #     center = getClusters(tile.worldSpots, tile.filled, 1)  # center is just x and y
        #     # pick random tile, 1-18 -> 0-17 inclusive
        #     # tileIndex = random.sample(range(0, 18), 1)[0]
        #
        #     tileIndex = addFloorTiles[random.sample(range(0,6), 1)[0]] # choose from subset
        #     newTile = getTile(tileIndex)
        #     # set tile properties - add noise to x and y, make sure it's 3D
        #     newTile.origin = np.append(np.add(center, [random.randint(-3, 4), random.randint(-3, 4)]), getHeight(numLevels))  # add noise, up to 3 mm in any dir
        #     newTile.rotation = getRotation(newTile, [newTile.origin], tile.worldSpots)
        #     rotate = getRotateMat(newTile.rotation)
        #     for spot in newTile.spots:
        #         tempSpot = tupleTransform(spot, newTile.origin, rotate)
        #         newTile.worldSpots.append(np.append(tempSpot, getHeight(numLevels)))  # include height in the worldSpots array
        #     newTile.filled = np.zeros(len(newTile.spots)) # empty to start
        #     tempHistory = self.history + [[1, newTile]]
        #     # simulate old tower with addons
        #     collapse = CoppeliaTower.simulate(self.oldtower, tempHistory)
        #     if not collapse:
        #         self.history = tempHistory
        #         np.append(self.modifiedtower, [newTile])
        #         return False, -5  # 15  # added, keep going (additional reward for risky move)
        #     # if made it here, didn't work, reset worldSpots? Should have been taken care of already but...
        #     newTile.worldSpots = []
        # return True, -4  # 30 # didn't work, end episode, consider a success?

    # based on state, check if any legal moves left. if so, return 0, else return 1
    def checkEnd(self):
        tile = self.modifiedtower[-1][0]
        count = 0
        for spot in tile.spots:
            # check if spot is empty and colour is available
            if not tile.filled[count] and tile.colours[count] in self.myPillars:
                # at least 1 available spot
                return False
            count += 1
        # print("End Episode")
        return True  # no more spots

    def step2(self, action):
        # just check that the action is legal
        self.state = self.getState()
        location = self.getCoord(action)
        self.count += 1
        print("Action:", action)
        print("State:", self.state)
        if location is None:
            print("Illegal")
            reward = -15
        else:
            # is legal, swap myPillar and fill
            # myPillar changes in getCoord
            reward = 15
            self.modifiedtower[-1][0].filled[action] = 1
            self.state = self.getState()
            print("New State:", self.state)

        endflag = self.checkEnd()

        if self.count > 20:
            endflag = 1

        # if self.card == 0:
        #     endflag = 1
        #     print("Success!!!")

        info = dict()
        print()

        return self.state, reward, endflag, info

    def step(self, action):

        location = self.getCoord(action)  # get coordinate of chosen spot - must be local
        self.count += 1
        endflag = 0
        tempReward = 0
        # print("State:", self.state)
        print("Action:", action)

        # shouldn't happen
        # modify tower to reflect filled (must do after simulation so that simulated in order)
        if action == 5:  # choose to pass, large negative reward
            # action = random.randint(0,4)
            reward = -80
            endflag = 1
            print("Chose to pass")

        elif location is None:  # not valid move
            print("Illegal!")
            self.illegalMoves += 1
            reward = -15  # negative reward, no change in state, return
        else:  # only modify tower if mover is valid/successful

            tile = self.modifiedtower[-1][0]
            self.history.append([0, location, tile.id])  # 0 means it's a pillar object

            # instead of simulating, just use hardcode
            # build tower so that it stands, then add pillar. Observe collapse
            collapse = CoppeliaTower.simulate(self.oldtower, self.history)
            # collapse = 0  # didn't collapse


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
            # self.recordTowers.append(self.getFullState())
            # self.card -= 1
            if collapse: # or self.card == 0:
                endflag = 1
            else:
                endflag, tempReward = self.addFloor()  # if finished 3rd level, success (endFlag = 1) else add next tile
            # temp reward gives reward for cases where ending without card == 0

            reward = self.getReward(collapse)
            reward += tempReward  # from adding floor
             # finished episode

            # if self.card == 0 and not collapse:
            #     # reward += 60
            #     reward += 30
            #     print("Success!")

            reward = 15  # overwrite to simplify

        info = dict()
        self.state = self.getState()
        if self.checkEnd():
            endflag = 1
            # reward += 20  # did good job, nothing left to do

        if endflag:
            self.count = 0
        # print(action)
        if self.count > 20 and endflag == 0:
            # reward = 0
            endflag = 1  # give up
            print("Give up!!")


        print("Reward:", reward)
        # print("New State:", self.state)
        # print("State:", self.state)
        print()
        return self.state, reward, endflag, info

    def getIllegal(self):
        return self.recordIllegal

    def reset(self):
        self.getNext()  # assigns tower

        self.modifiedtower = self.copy(self.oldtower)
        # new set of pillars
        self.myPillars = self.getPillars(5)
        # self.myPillars = [1, 1, 2, 2, 3]

        while self.checkEnd():
            self.myPillars = self.getPillars(5)
            print("Re-drawing Pillars")

        # self.campPillars = self.getPillars(6)
        # self.partnerPillars = self.getPillars(5)
        # self.card = 6  #random.randrange(1, 4)  # random, 1, 2, or 3 = # of pillars to place
        self.count = 0
        self.history = []
        self.recordIllegal.append(self.illegalMoves)
        self.illegalMoves = 0

        # self.state = 0
        self.state = self.getState()
        return np.float32(self.state)

