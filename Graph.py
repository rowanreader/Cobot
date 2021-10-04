import numpy as np


# for pillar spots
class Node:
    def __init__(self, nodeID, floor, x, y, colour, weight=0, filled=False):
        self.id = nodeID
        self.parent = floor
        self.x = x
        self.y = y
        self.colour = colour # number representing the colour
        self.weight = weight
        self.filled = filled

    def addPillar(self, weight):
        self.weight = weight
        self.filled = True


# floor object, has nodes on it
class Floor:
    # needs id of floor, x and y coords of center (defined for each floor), level, represented by int,
    # list of nodes on floor, center of mass, pillars of support (nodes)
    # to initialize floor (as not in tower, or at base) don't need pos
    def __init__(self, floorID, x, y, level, nodes, com, pos=None):
        self.id = floorID
        # x and y are where the origin of the floor is placed
        self.x = x
        self.y = y
        self.level = level
        self.nodes = nodes
        self.com = com
        self.pos = pos

    # when pillar is added, need to adjust com
    def recalculateCoM(self):
        sumX = 0
        sumY = 0
        weights = 0
        # find weighted average based on weights and coords
        for i in self.nodes:
            sumX += i.x * i.weight
            sumY += i.y * i.weight
            weights += i.weight

        self.com = [sumX/weights, sumY/weights]

    # based on com and pos, is it stable
    # need to fit shape to points in pos
    def balanced(self):
        if self.pos is None:  # floor is either a base - always balanced, or not in tower
            return True


# to build an actual model of the tower
class Model:
    # initialFloors is a list of the initial floors (level 0), level keeps track of the height of the tower
    def __init__(self, initialFloors, level=0, maxLevel=5):
        self.floors = np.array(initialFloors)
        self.level = level
        self.maxLevel = maxLevel

    # takes in floor, coordinates of floor
    def addFloor(self, floor, x, y, level):
        # identify pillars of support, add to floor variables

        # check if level is higher now
        pass

    # take in node, floor, and colour of pillar and pillar weight (should be standard)
    def addPillar(self, floor, node, colour, pillarWeight):
        # check that colour is appropriate
        if colour != node.colour:
            return False

        node.filled = True

        # for each floor it's connected to, recursively distribute the weight
        # use pos of the floor, recursively add
        self.recalculateWeight(floor, node, pillarWeight)

    def recalculateWeight(self, floor, node, pillarWeight):
        node.weight += pillarWeight

        pillars = floor.pos
        if pillars is None:
            return

        # weight is equally distributed amongst all pillars
        #### MAY HAVE TO CHANGE TO WEIGHTED PROPORTIONS
        newWeight = pillarWeight/len(pillars)
        for i in pillars:
            self.recalculateWeight(i.parent, i, newWeight)
        # have recalculated weight for all
        return

    def increaseMaxLevel(self):
        self.maxLevel += 1






