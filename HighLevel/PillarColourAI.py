import random
from HighTowerSim import getTile
import numpy as np

# very simple agent, doesn't require tower sim at all
# reward based on how many pillars have their own unique place

class Colour:

    def __int__(self):

        self.campPillars = self.getPillars(6)
        self.myPillars = self.getPillars(5)

        self.allPillars = self.campPillars + self.myPillars
        # choose a single random tile
        self.tileId = random.randint(0,17)
        self.tile = getTile(self.tileId)

        # 0 represents ghost
        self.colours = np.zeros(5)
        count = 0
        for i in self.tile.colours:
            self.colours[count] = i

        # combine state into list of 16
        self.state = self.campPillars + self.pillars + self.colours

    def getPillars(self, num):
        choices = [1, 2, 3, 4, 5]
        freq = [0.39, 0.26, 0.16, 0.11, 0.08]
        return random.choices(choices, freq, k=num)


    # action is 5 element array of chosen colours
    # check how many pillars have a unique spot to go to
    def getReward(self, action):
        count = 0
        spots = np.copy(self.colours)
        for i in action:
            if i in spots:
                count += 1
                spots = np.delete(spots, np.argwhere(spots == i))
        return count


    # action is an 11 element array of probabilities?
    # take top 5, identify cooresponding pillar colours, get reward
    def step(self, action):
        # gets indexes of top 5 values
        modifiedActInd = np.argpartition(action, -5)[-5:]
        # get colours
        modifiedAct = self.allPillars[modifiedActInd]
        reward = self.getReward(modifiedAct)
        endflag = 1  # endflag is always 1
        info = dict()

        return self.state, reward, endflag, info