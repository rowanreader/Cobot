import numpy as np


# for pillar spots
class Node:
    # floor needs to be id?
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
    # needs id of floor, x and y coords of center (defined for each floor), level, represented by int, orientation
    # list of nodes on floor, center of mass, pillars of support (nodes)
    # to initialize floor (as not in tower, or at base) don't need pos
    def __init__(self, floorID, x, y, level, ori, nodes, com=[0,0], pos=None):
        self.id = floorID
        # x and y are where the origin of the floor is placed
        self.x = x
        self.y = y
        self.level = level
        self.ori = ori
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

    def getNode(self, id):
        for i in self.nodes:
            if i.id == id:
                return i

# to build an actual model of the tower
class Model:
    # initialFloors is a list of the initial floors (level 0), level keeps track of the height of the tower
    # maxLevel is condition for winning
    def __init__(self, initialFloors, level=0, maxLevel=5):
        self.model = [[initialFloors]] # should be a 3D array (initialFloors is already an array)
        self.level = level
        self.maxLevel = maxLevel

    # takes in floor, adds to model. Floor should have been initialized with x, y, orientation etc
    # assume floors themselve have negligible weight, don't need to recalculate
    def addFloor(self, floor, x,y, level, ori):
        # identify pillars of support, add to floor variables
        floor.x = x
        floor.y = y
        floor.level = level
        floor.ori = ori
        if level > self.level:
            self.level = level
            self.model.append([floor])
        else:
            self.model[level].append(floor)
        # check if level is higher now


    # take in node, floor, and colour of pillar and pillar weight (should be standard)
    def addPillar(self, floorID, nodeID, colour, pillarWeight):
        floor = self.getFloor(floorID)
        node = floor.getNode(nodeID)
        # check that colour is appropriate
        if colour != node.colour:
            print("wrong colour")
            return False

        if node.filled == True:
            print("already has a pillar")
        else:
            node.filled = True

        # for each floor it's connected to, recursively distribute the weight
        # use pos of the floor, recursively add
        self.recalculateWeight(floor, node, pillarWeight)

    def recalculateWeight(self, floor, node, pillarWeight):
        node.weight += pillarWeight
        # get pillars of support
        pillars = floor.pos
        if pillars is None or pillarWeight < thresh: # base case
            return

        # weight is equally distributed amongst all pillars
        #### MAY HAVE TO CHANGE TO WEIGHTED PROPORTIONS
        newWeight = pillarWeight/len(pillars) # get number of pillars supporting, assume each takes equal burden
        for i in pillars:
            self.recalculateWeight(self.getFloor(i.parent), i, newWeight)
        # have recalculated weight for all
        # recalculate com
        floor.recalculateCoM()
        return

    def getFloor(self, floorID):
        for i in Floors:
            if i.id == floorID:
                return i

    def increaseMaxLevel(self):
        self.maxLevel += 1


# paramaters, initialize
thresh = 0.001 # in grams

# floors
# ori is degrees off from defined neutral position (which all nodes are measured from)
# x and y are in mm
# Floor: floorID, x, y, level, ori, nodes, com=[0,0], pos=None
# Node: nodeID, floor, x, y, colour, weight=0, filled=False
# colors: black = 0, red = 1, blue = 2, yellow = 3, white = 4
floor1 = Floor(1, 10, 20, 0, 20, [Node(1, 1, -31, 16, 3), Node(2, 1, 0, 28, 1), Node(3, 1, 34, 16, 4)]) # crescent moon
floor2 = Floor(2, 20, 30, 0, 30, [Node(1, 2, 0, 61, 0), Node(2, 2, 52, 40, 4), Node(3, 2, 52, 40, 3)]) # triskeleton
floor3 = Floor(3, 30, 60, 0, 40, [Node(1, 3, -11, 37, 0), Node(2, 3, 34, 38, 0), Node(3, 3, 33, -17, 2), Node(4, 3, -3, -36, 3), Node(5, 3, -32, -38, 3)]) # orchid

floor4 = Floor(4, 0, 0, 0, 30, [Node(1, 4, -37, -7, 1), Node(2, 4, -9, -20, 4), Node(3, 4, 35, -11, 3), Node(4, 4, 20, 27, 3)]) # toucan
initialFloors = [floor1, floor2, floor3]
Floors = [floor1, floor2, floor3, floor4]
pillarWeight = 5 # in grams
tower = Model(initialFloors)

# floor, node, colour, pillarWeight
tower.addPillar(1, 3, 4, pillarWeight)
tower.addFloor(floor4, 10, 20, 1, 20)
print(tower.level)