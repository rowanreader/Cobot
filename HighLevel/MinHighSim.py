# for generating files

# the same as TowerSim but with different outputs:
# [((x, y, z, theta) (x, y, z, filled) * 5) *3] * 3
# origin of floor + angle, all points on that floor in global coords, 1 array/level
import random
import numpy as np
import sys
import time
import pickle
import cdd as pcdd
from scipy.spatial import ConvexHull
import matplotlib.pyplot as plt
import SawyerSim
import PhysicsConnect
from stable_baselines3 import SAC


printstuff = False
rad = 10


# spots have diameter = 20 mm, rad = 10 mm
# pillars have diameter = 12 mm, rad = 6 mm
# pillars have height of 45 mm

# given a coordinate according to local origin, world origin (where (0,0) is in world coords) and rotation matrix,
# transform coordinate from local system to world
# rotate should be 2x2 since working in 2D
def tupleTransform(t1, origin, rotate):
    if len(t1) != len(origin):
        print("Tupples are not the same size")
        return ()
    t1 = np.array(t1)
    origin = np.array(origin)
    newT = np.matmul(t1, rotate) + origin
    return newT


# check if these spots are too close (2 radii)
def checkCollide(t1, t2):
    x = t2[0] - t1[0]
    y = t2[1] - t1[1]
    dist = np.sqrt(x ** 2 + y ** 2)
    if dist >= 2 * rad:  # no collision
        return False
    else:  # collision
        return True


def getRotateMat(angle):
    rotate = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    return rotate


# gets distance between p1 and p2
def getDist(p1, p2):
    x = p1[0] - p2[0]
    y = p1[1] - p2[1]
    dist = np.sqrt(x ** 2 + y ** 2)
    return dist

def getHeight(level): # level is 0 1 or 2
    height = 45  # pillars in mm
    tileHeight = 2  # thickness of cardboard
    return (level * height + tileHeight * (level + 1))

# given a point an a list of clusters, finds nearst

def closestCluster(point, clusters):
    min = sys.maxsize
    count = 0
    for k in clusters:
        dist = getDist(point, k)
        if dist < min:
            min = dist
            cluster = count
        count += 1
    return cluster



# takes in coords and id of new tile, uses tile placement AI to determine origin
def getOrigin(spots, id):
    state = np.zeros(11)
    index = 0
    for spot in spots:
        state[index:index+2] = spot
        index += 2

    state[10] = id
    state = [x/50 for x in state]
    # based on local coords
    model = SAC.load("SAC1000_TilePlace_best/best_model")
    action, _states2 = model.predict(state, deterministic=False)
    return action

# takes in list of coordinates, binary list representing which are filled, and the number of clusters to find, finds most balanced set

def getClusters(spots, filled, numClusters):
    alpha = 0.1

    num = len(spots)
    pillars = []
    for k in range(num):
        if filled[k] == 1:
            pillars.append(spots[k])

    maxX = int(max(pillars, key=lambda x: x[0])[0])
    maxY = int(max(pillars, key=lambda x: x[1])[1])
    minX = int(min(pillars, key=lambda x: x[0])[0])
    minY = int(min(pillars, key=lambda x: x[1])[1])

    if numClusters > len(range(minX, maxX)):  # sometime all spots have the same x or y coord
        if printstuff:
            print("XRange too small for clusters, defaulting to minX")
        xs = [minX] * numClusters
    else:
        xs = random.sample(range(minX, maxX), numClusters)

    if numClusters > len(range(minY, maxY)):
        if printstuff:
            print("YRange too small for clusters, defaulting to minY")
        ys = [minY] * numClusters
    # randomly place numCluster points within the bounds of min/max
    else:
        ys = random.sample(range(minY, maxY), numClusters)

    origin = []
    for k in range(numClusters):
        origin.append([xs[k], ys[k]])

    thresh = 5
    error = thresh + 1
    count = 0
    # iteratively for each pillar, pull the cluster center closest to the pillar closer
    while error > thresh and count < 100:
        error = 0
        for k in pillars:
            nearest = closestCluster(k, origin)  # index of origin that is closest to the point
            xDist = k[0] - origin[nearest][0]  # distance between
            yDist = k[1] - origin[nearest][1]
            # move proportionally to distance
            origin[nearest][0] += alpha * xDist
            origin[nearest][1] += alpha * yDist

            error += np.sqrt(alpha * xDist ** 2 + alpha * yDist ** 2)
        count += 1
    # if count == 100:
    #     print("100 iterations of cluster adjustment!")
    #     print("Error: " + str(error))
    if printstuff == True:
        print("Cluster error: " + str(error))
    # print(origin)
    # print(spots)
    # plt.scatter([x[0] for x in spots], [y[1] for y in spots], color='blue')
    # plt.scatter(origin[0][0], origin[0][1], color='red')
    # plt.show()
    # print(error)
    # print(count)

    ### for validation set only
    # origin[nearest][0] += 35

    return origin

    # check error size, once less than threshold (or number of iterations is too large), stop

    # check that there is one x coord larger and 1 x coord smaller. similar for y. if not, print error

class Tile:
    # spots is array of tupples, contatining distance from com of tile in x and y (mm)
    # rotation is angle of rotation the tile is in (radians)
    def __init__(self, Id, spots, rotation=0, origin=(0, 0, 0), outline=[(0,0)], colours=()):
        self.id = Id
        self.spots = spots
        self.rotation = rotation
        self.origin = origin
        self.worldSpots = [] # based on rotation and level
        self.filled = [] # binary array of filled or not filled, corresponding to worldSpots
        self.colours = colours
        self.outline = outline

# list of all tiles and their spots relative to their origin (mm)
# only ints allowed
tile01 = Tile("Tile01", [(-3, 47), (20, 10), (-1, -42)], outline=[(20, 90), (35, 8), (6, -90), (-26, -28), (0, 30), (-23, 30)], colours=(2, 1, 2))  # (R, Y, R)
# Majora's Mask
tile02 = Tile("Tile02", [(27, 10), (-5, -26), (-22, 17)], outline=[(0, 68), (62, 25), (43, -50), (-40, -50), (-60, 24)], colours=(3, 3, 2))  # (K, K, R)
# Shard
tile03 = Tile("Tile03", [(-61, -15), (-5, 13), (16, -18), (55, -3)], outline=[(0, 25), (94, -15), (13, -40), (-16, -16), (-60, -22), (-104, -5)], colours=(2, 1, 2, 4))  # (R, Y, R, W)
# Square Spiral
tile04 = Tile("Tile04", [(-34, 45), (-11, -10), (13, 10), (35, -45)], outline=[(-45, 60), (57, 43), (45, -65), (-55, -43)], colours=(1, 3, 3, 1))  # (Y, K, K, Y)
# Toucan
tile05 = Tile("Tile05", [(-34, -13), (-6, -22), (23, 25), (34, -11)], outline=[(85, 17), (10, 60), (-25, 5), (-57, 15), (-55, 43), (-78, -14), (0, -54), (70, -30)], colours=(2, 3, 1, 1))  # (R, K, Y, Y)
# Crescent
tile06 = Tile("Tile06", [(-31, 19), (2, 33), (34, 16)], outline=[(43, -57), (50, 10), (0, 35), (-52, 6), (-40, -60), (-35, -15), (0, 3), (35, 15)], colours=(1, 2, 4))  # (Y, R, W)
# Half Ax Head
tile07 = Tile("Tile07", [(-26, -25), (2, 34), (11, -20)], outline=[(0, 60), (-25, 31), (-27, -15), (-68, -14), (0, -45), (67, -12), (28, 38)], colours=(1, 1, 1))  # (Y, Y, Y)
# Ax
tile08 = Tile("Tile08", [(-75, 26), (-40, -17), (-35, 38), (83, -23)], outline=[(-60, -27), (135, 15), (-7, -2), (10, 40), (25, 48), (-43, 62), (-72, 14)], colours=(1, 3, 3, 1))  # (Y, K, K, Y)
# Triskelion
tile09 = Tile("Tile09", [(-53, -47), (-1, 61), (55, -47)], outline=[(-15, 85), (-70, -55), (-81, 33)], colours=(3, 1, 4))  # (K, Y, W)
# Orchid
tile10 = Tile("Tile10", [(-33, -34), (-13, 38), (-3, -35), (31, -19), (39, 39)], outline=[(60, 65), (40, 11), (53, -14), (-17, -53), (-98, 12), (-4, -3), (-28, 17), (-41, 65), (10, 35)], colours=(1, 3, 1, 5, 3))  # (Y, K, Y, B, K)
# Red Square
tile11 = Tile("Tile11", [(-24, 28), (-13, -38), (28, -39), (32, 31)], outline=[(19, 70), (83, 19), (35, -55), (10, -43), (-51, -46), (-34, 0), (-52, 35), (62, 15)], colours=(2, 2, 2, 2))  # (R, R, R, R)
# Massive
tile12 = Tile("Tile12", [(-65, 0), (-39, 0), (1, -39), (1, 39), (41, 0)], outline=[(0, 79), (45, 30), (110, 0), (53, -30), (0, -80), (-38, -33), (-110, 0), (-43, 33)], colours=(4, 1, 2, 5, 1))# (W, Y, R, B, Y)
# Square
tile13 = Tile("Tile13", [(-11, -12), (-11, 11), (11, 11), (45, 39), (42, -42)], outline=[(65, 67), (40, 0), (70, -65), (0, 40), (-64, -67), (-40, 0), (-67, 64), (0, 39)], colours=(2, 4, 2, 4, 1))  # (R, W, R, W, Y)
# Ax Head
tile14 = Tile("Tile14", [(-15, 37), (0, 0), (14, 38)], outline=[(42, 32), (14, 0), (41, -35), (0, -55), (-44, -34), (-16, 0), (-41, 35), (0, 54)], colours=(1, 2, 5))  # (Y, R, B)
# Flame
tile15 = Tile("Tile15", [(-17, 10), (-1, 22), (20, 10)], outline = [(40, 11), (12, -45), (0, -2), (-10, -20), (9, -68), (-36, 13), (0, 41)], colours=(1, 2, 1))  # (Y, R, Y)
# Shuriken
tile16 = Tile("Tile16", [(-24, -11), (0, 0), (1, 26), (22, -16)], outline=[(37, 22), (30, -35), (0, -45), (-44, -10), (-40, 20), (15, 41)], colours=(1, 3, 2, 1))  # (Y, K, R, Y)
# Stairs
tile17 = Tile("Tile17", [(-60, -8), (-22, 28), (16, -3), (60, -6)], outline=[(45, 40), (75, 18), (80, -53), (0, 10), (-80, -53), (-72, 18), (-43, 30)], colours=(1, 1, 2, 1))  # (Y, Y, R, Y)
# Spaceship
tile18 = Tile("Tile18", [(-26, 32), (0, 9), (26, 32)], outline=[(55, 46), (13, -26), (7, -87), (-10, -30), (-53, 43), (0, 30)], colours=(2, 3, 1))  # (R, K, Y)
# tiles = [tile09, tile19, tile20]
tiles = [tile01, tile02, tile03, tile04, tile05, tile07, tile08, tile10, tile11, tile12, tile13, tile14,
         tile15, tile16, tile17, tile18]

tiles1 = [tile10, tile12, tile05, tile11, tile08] # tile09, tile06
tiles2 = [tile07, tile01, tile03, tile18, tile13, tile04]
tiles3 = [tile14, tile17, tile02, tile16, tile15]

# get function, returns tile object based on index
def getTile(num):
    return tiles[num]
# takes in array of coordinates and binary array indicating whether they are occupied (1) or not (0)
# returns all spots that are occupied

def getOccupied(spots, filled):
    occupied = []
    count1 = 0
    for j in filled:  # goes up to 3
        temp = []
        count2 = 0
        num = len(j)
        for i in range(num):
            if j[i] == 1:
                # spotsTemp = np.append(spots[count1][count2], count1)
                spotsTemp = spots[count1][count2]
                temp.append(spotsTemp)
            count2 += 1
        occupied.append(temp)
        count1 += 1
    if printstuff:
        print("Occupied:")
        print(occupied)
    return occupied


def getRotation(tile, origin, pillars):
    # iterate through rotations, find max area
    maxArea = 0
    maxRotation = 0

    # gotta add z component, just set to 0
    newTile = []
    newPillars = []
    for i in tile.outline:
        newTile.append([i[0], i[1], 0])
    for i in pillars:
        newPillars.append(np.append(i, 0))

    num = len(newTile)
    # every 5 degrees
    for rotation in range(0, 360, 5):
        # rotate tile points by i degrees
        rad = np.deg2rad(rotation)
        mat = getRotateMat(rad)
        rotatedTile = np.empty([num, 3]) # num rows, 3 columns (x, y ,z)
        for i in range(num):
            # rotate and translate
            temp = np.matmul(newTile[i][0:2], mat) + [origin[0][0], origin[0][1]]
            rotatedTile[i] = np.append(temp, 0)

        # get intersect area
        area = getHull(rotatedTile, newPillars)
        if area > maxArea:
            maxArea = area
            maxRotation = rotation

    print(maxRotation)
    return maxRotation


def getHull(newTile, newPillars):
    try:
        numT = len(newTile)
        numP = len(newPillars)
        # make the V-representation of the first cube; you have to prepend
        # with a column of ones
        v1 = np.column_stack((np.ones(numT), newTile))
        # v1 = np.column_stack((np.ones(8), pts1))
        mat = pcdd.Matrix(v1, number_type='fraction')  # use fractions if possible
        mat.rep_type = pcdd.RepType.GENERATOR
        poly1 = pcdd.Polyhedron(mat)

        # make the V-representation of the second cube; you have to prepend
        # with a column of ones
        v2 = np.column_stack((np.ones(numP), newPillars))
        # v2 = np.column_stack((np.ones(8), pts2))
        mat = pcdd.Matrix(v2, number_type='fraction')
        mat.rep_type = pcdd.RepType.GENERATOR
        poly2 = pcdd.Polyhedron(mat)

        # H-representation of the first cube
        h1 = poly1.get_inequalities()

        # H-representation of the second cube
        h2 = poly2.get_inequalities()

        # join the two sets of linear inequalities; this will give the intersection
        hintersection = np.vstack((h1, h2))

        # make the V-representation of the intersection
        mat = pcdd.Matrix(hintersection, number_type='fraction')
        mat.rep_type = pcdd.RepType.INEQUALITY
        polyintersection = pcdd.Polyhedron(mat)

        # get the vertices; they are given in a matrix prepended by a column of ones
        vintersection = polyintersection.get_generators()

        # get rid of the column of ones
        num = len(vintersection)
        ptsintersection = np.array([vintersection[i][1:4] for i in range(num)])

        # these are the vertices of the intersection; it remains to take
        # the convex hull
        hull = ConvexHull(ptsintersection)
        # plotHull(newTile, newPillars, ptsintersection)
        return hull.volume # for 2D area is perimeter and volume is area
    except Exception as e:
        return 0 # just say bad result


def plotHull(hull1, hull2, intersect):
    plt.plot(hull1[:,0], hull1[:,1], color="blue")
    plt.plot([x[0] for x in hull2], [y[1] for y in hull2], color="green")
    plt.plot([float(x) for x in intersect[:, 0]], [float(x) for x in intersect[:, 1]], color="red")

    plt.show()


# Just the info so you can reinstantiate
# Dagger 01
tileData01 = ("Tile01", [(-3, 47), (20, 10), (-1, -42)], [(20, 90), (35, 8), (6, -90), (-26, -28), (0, 30), (-23, 30)], (2, 1, 2))  # (R, Y, R)
# Majora's Mask
tileData02 = ("Tile02", [(27, 10), (-5, -26), (-22, 17)], [(0, 68), (62, 25), (43, -50), (-40, -50), (-60, 24)], (3, 3, 2))  # (K, K, R)
# Shard
tileData03 = ("Tile03", [(-61, -15), (-5, 13), (16, -18), (55, -3)], [(0, 25), (94, -15), (13, -40), (-16, -16), (-60, -22), (-104, -5)], (2, 1, 2, 4))  # (R, Y, R, W)
# Square Spiral
tileData04 = ("Tile04", [(-34, 45), (-11, -10), (13, 10), (35, -45)], [(-45, 60), (57, 43), (45, -65), (-55, -43)], (1, 3, 3, 1))  # (Y, K, K, Y)
# Toucan
tileData05 = ("Tile05", [(-34, -13), (-6, -22), (23, 25), (34, -11)], [(85, 17), (10, 60), (-25, 5), (-57, 15), (-55, 43), (-78, -14), (0, -54), (70, -30)], (2, 3, 1, 1))  # (R, K, Y, Y)
# Crescent
tileData06 = ("Tile06", [(-31, 19), (2, 33), (34, 16)], [(43, -57), (50, 10), (0, 35), (-52, 6), (-40, -60), (-35, -15), (0, 3), (35, 15)], (1, 2, 4))  # (Y, R, W)
# Half Ax Head
tileData07 = ("Tile07", [(-26, -25), (2, 34), (11, -20)], [(0, 60), (-25, 31), (-27, -15), (-68, -14), (0, -45), (67, -12), (28, 38)], (1, 1, 1))  # (Y, Y, Y)
# Ax
tileData08 = ("Tile08", [(-75, 26), (-40, -17), (-35, 38), (83, -23)], [(-60, -27), (135, 15), (-7, -2), (10, 40), (25, 48), (-43, 62), (-72, 14)], (1, 3, 3, 1))  # (Y, K, K, Y)
# Triskelion
tileData09 = ("Tile09", [(-53, -47), (-1, 61), (55, -47)], [(-15, 85), (-70, -55), (-81, 33)], (3, 1, 4))  # (K, Y, W)
# Orchid
tileData10 = ("Tile10", [(-33, -34), (-13, 38), (-3, -35), (31, -19), (39, 39)], [(60, 65), (40, 11), (53, -14), (-17, -53), (-98, 12), (-4, -3), (-28, 17), (-41, 65), (10, 35)], (1, 3, 1, 5, 3))  # (Y, K, Y, B, K)
# Red Square
tileData11 = ("Tile11", [(-24, 28), (-13, -38), (28, -39), (32, 31)], [(19, 70), (83, 19), (35, -55), (10, -43), (-51, -46), (-34, 0), (-52, 35), (62, 15)], (2, 2, 2, 2))  # (R, R, R, R)
# Massive
tileData12 = ("Tile12", [(-65, 0), (-39, 0), (1, -39), (1, 39), (41, 0)], [(0, 79), (45, 30), (110, 0), (53, -30), (0, -80), (-38, -33), (-110, 0), (-43, 33)], (4, 1, 2, 5, 1))# (W, Y, R, B, Y)
# Square
tileData13 = ("Tile13", [(-11, -12), (-11, 11), (11, 11), (45, 39), (42, -42)], [(65, 67), (40, 0), (70, -65), (0, 40), (-64, -67), (-40, 0), (-67, 64), (0, 39)], (2, 4, 2, 4, 1))  # (R, W, R, W, Y)
# Ax Head
tileData14 = ("Tile14", [(-15, 37), (0, 0), (14, 38)], [(42, 32), (14, 0), (41, -35), (0, -55), (-44, -34), (-16, 0), (-41, 35), (0, 54)], (1, 2, 5))  # (Y, R, B)
# Flame
tileData15 = ("Tile15", [(-17, 10), (-1, 22), (20, 10)], [(40, 11), (12, -45), (0, -2), (-10, -20), (9, -68), (-36, 13), (0, 41)], (1, 2, 1))  # (Y, R, Y)
# Shuriken
tileData16 = ("Tile16", [(-24, -11), (0, 0), (1, 26), (22, -16)], [(37, 22), (30, -35), (0, -45), (-44, -10), (-40, 20), (15, 41)], (1, 3, 2, 1))  # (Y, K, R, Y)
# StairsTile
tileData17 = ("Tile17", [(-60, -8), (-22, 28), (16, -3), (60, -6)], [(45, 40), (75, 18), (80, -53), (0, 10), (-80, -53), (-72, 18), (-43, 30)], (1, 1, 2, 1))  # (Y, Y, R, Y)
# Spaceship
tileData18 = ("Tile18", [(-26, 32), (0, 9), (26, 32)], [(55, 46), (13, -26), (7, -87), (-10, -30), (-53, 43), (0, 30)], (2, 3, 1))  # (R, K, Y)

tilesData = [tileData01, tileData02, tileData03, tileData04, tileData05, tileData06, tileData07, tileData08,
             tileData10, tileData11, tileData12, tileData13, tileData14, tileData15, tileData16, tileData17, tileData18]
tempTowers = [13, 12]
shift = [[], [0, 0], [0, 0]]
rots = [0, 30, 70]
# builds random tower
# returns occupied spots and origin of tiles
# can make plane based on that

def build():
    taken = []  # holds already used tiles
    allWorldSpots = []
    towerTiles = []
    allFilled = []
    # randomly choose 3 tiles and place in fixed origin spots
    temp1 = len(tiles)
    firstFloorId = random.sample(range(0, temp1), 1)  # corresponds to index in tiles, so number 0 = tile01

    # firstFloorId = [tempTowers[0]]


    # print(firstFloorId)
    taken += firstFloorId
    # firstFloorId = [2, 11, 15]
    data = tilesData[firstFloorId[0]]
    # must reinstantiate
    firstFloors = [Tile(data[0], data[1], outline=data[2], colours=data[3])]
    # firstFloors = [tiles[firstFloorId[0]]]
    if printstuff == True:
        print(firstFloorId)
    origin1 = [[0, 0], [600, 300], [700, 0]]  # mm from origin of world coord
    count = 0
    for i in firstFloors:
        i.origin = np.append(origin1[count], getHeight(0))
        count += 1

    origins = [origin1]

    # randomly choose level (1, 2, 3)
    # levels = random.randint(1, 3)  # either 1, 2 or 3
    choices = [1, 2, 3]
    freq = [0.1, 0.4, 0.5] # frequency of choosing levels
    levels =  random.choices(choices, freq, k=1)[0]


    # print(levels)
    # levels = 2
    # levels = 1
    if printstuff == True:
        print("Going up to level " + str(levels))
    filledSpots = []

    # always going to have 1st level at least
    # choose configuration such that pillar spots aren't on top of each other
    # figure out spots - first can stay where it is, adjust 2nd and 3rd around it
    # assuming no rotation, where would spots be in world coordinates
    firstSpotsWorld = []
    rotate = np.identity(2)
    angle = 0
    for spot in firstFloors[0].spots:
        tempSpot = tupleTransform(spot, origin1[0], rotate)
        firstSpotsWorld.append(tempSpot)  # gonna switch to arrays instead of tuples
        firstFloors[0].worldSpots.append(np.append(tempSpot, getHeight(0))) # include height in the worldSpots array

    firstFloors[0].rotation = 0 # set rotation of 1st tile of floor

    allWorldSpots.append(firstSpotsWorld)

    if printstuff == True:
        print("Spots level 1: " + str(firstSpotsWorld))
    # randomly fill 1st level with pillars (dependent on level chosen -> higher = more)
    # for level 1: 0 - 100%, level 2: 50-100%, level 3: 80-100%

    numPillars1 = len(firstSpotsWorld)
    filled1 = np.zeros(numPillars1)  # binary, either filled spot or not

    # here 1 refers to the level number, not the tile number
    if levels == 1:
        # pick number between 0 and the number of pillars
        numChosen1 = random.sample(range(0, numPillars1-1), 1)[0]

    elif levels == 2:
        # numChosen1 = random.sample(range(numPillars1 // 2 + 1, numPillars1), 1)[0]  # limit the number of pillars to be between half and all
        numChosen1 = numPillars1
    elif levels == 3:
        # numChosen1 = random.sample(range(int(numPillars1 * 0.8) - 1, numPillars1), 1)[0]  # limit the number of pillars to be between half and all
        numChosen1 = numPillars1
    # numChosen1 = numPillars1  # select all
    chosen1 = random.sample(range(0, numPillars1), numChosen1)

    # set chosen pillars to 1 (filled)
    for i in chosen1:
        filled1[i] = 1
    # requires 2D array
    occupied1 = getOccupied([firstSpotsWorld], [filled1])
    filledSpots.append(occupied1)
    allFilled.append(filled1)

    # separate out into arrays for each tile of level
    count = 0
    for i in firstFloors:
        numSpots = len(i.spots)
        i.filled = filled1[count:numSpots+count]
        count += numSpots

    towerTiles.append(firstFloors)
    # now build second level (if applicable)
    if levels > 1:

        # for second floor either 2 or 3 tiles
        # numTiles2 = random.sample(range(2, 4), 1)[0]
        # numTiles2 = 1
        # range2 = [x for x in range(0,18) if x not in taken]
        temp2 = len(tiles)
        secondFloorId = random.sample(range(0, temp2), 1)
        # print(secondFloorId)
        # secondFloorId = [tempTowers[1]]
        taken += secondFloorId
        if printstuff == True:
            print(secondFloorId)
        # secondFloors = []
        # for i in secondFloorId:
        #     secondFloors.append(tiles[i])
        data = tilesData[secondFloorId[0]]
        # must reinstantiate
        secondFloors = [Tile(data[0], data[1], outline=data[2], colours=data[3])]
        # origin2 = getClusters(firstSpotsWorld, filled1, 1)
        temp = getOrigin(firstFloors[0].spots, secondFloorId[0])
        origin2 = [[temp[0], temp[1]]]
        origin2[0][0] += shift[1][0]
        origin2[0][1] += shift[1][1]

        # print(origin2)
        # origin2 = origin1
        count = 0
        for i in secondFloors:
            i.origin = np.append(origin2[count], getHeight(1))
            count += 1
        origins.append(origin2)
        if printstuff == True:
            print("Level 2 origins: " + str(origin2))

        # given a tile, origin (centriod) and POS, find rotation of max overlap using convexhull
        # angle = getRotation(secondFloors[0], origin2, firstSpotsWorld)
        # angle += rots[1]
        angle = temp[2]
        secondFloors[0].rotation = angle  # this is actually the most recent angle from 1st floor
        rotate = getRotateMat(angle)
        # choose origin in center of triangle of pillars??? Given pillars and number of 'clusters' find centroids
        # orient so no collisions
        secondSpotsWorld = []
        # place 1st floor of 2nd level in neutral
        for spot in secondFloors[0].spots:
            tempSpot = tupleTransform(spot, origin2[0], rotate)
            secondSpotsWorld.append(tempSpot)
            secondFloors[0].worldSpots.append(np.append(tempSpot, getHeight(1)))

        allWorldSpots.append(secondSpotsWorld)
        if printstuff == True:
            print("Spots level 2: " + str(secondSpotsWorld))

        # fill in spots
        numPillars2 = len(secondSpotsWorld)
        filled2 = np.zeros(numPillars2)

        if levels == 2:  # 0 to 100%
            numChosen2 = random.sample(range(1, numPillars2-1), 1)[0]  # limit the number of pillars to be between half and all

        elif levels == 3:  # 60% to 100%
            # numChosen2 = random.sample(range(int(numPillars2 * 0.6), numPillars2), 1)[0]  # limit the number of pillars to be between half and all
            numChosen2 = numPillars2 # select all
        chosen2 = random.sample(range(0, numPillars2), numChosen2)
        for i in chosen2:
            filled2[i] = 1

        count = 0
        for i in secondFloors:
            numSpots = len(i.spots)
            i.filled = filled2[count:numSpots+count]
            count += numSpots

        occupied2 = getOccupied([secondSpotsWorld], [filled2])
        filledSpots.append(occupied2)
        allFilled.append(filled2)

        towerTiles.append(secondFloors)
    # same as prev
    if levels > 2:
        # for third floor either 1 or 2 tiles
        # numTiles3 = random.sample(range(1, 3), 1)[0]
        # range3 = [x for x in range(0, 18) if x not in taken]
        temp3 = len(tiles)
        thirdFloorId = random.sample(range(0, temp3), 1)
        # thirdFloorId = [tempTowers[2]]
        if printstuff == True:
            print(thirdFloorId)
        # thirdFloors = []
        # for i in thirdFloorId:
        #     thirdFloors.append(tiles[i])
        data = tilesData[thirdFloorId[0]]
        # must reinstantiate
        thirdFloors = [Tile(data[0], data[1], outline=data[2], colours=data[3])]

        temp = getOrigin(secondFloors[0].spots, thirdFloorId[0])
        origin3 = [[temp[0], temp[1]]]
        # origin3 = getClusters(secondSpotsWorld, filled2, 1)
        origin3[0][0] += shift[2][0]
        origin3[0][1] += shift[2][1]

        # origin3 = origin2
        count = 0
        for i in thirdFloors:
            i.origin = np.append(origin3[count], getHeight(2))
            count += 1
        origins.append(origin3)
        if printstuff == True:
            print("Level 3 origins: " + str(origin3))
        # choose origin in center of triangle of pillars??? Given pillars and number of 'clusters' find centroids
        # orient so no collisions
        thirdSpotsWorld = []
        # angle = getRotation(thirdFloors[0], origin3, secondSpotsWorld)
        # angle += rots[2]
        angle = temp[2]
        thirdFloors[0].rotation = angle
        rotate = getRotateMat(angle)
        # place 1st floor of 3rd level in neutral
        for spot in thirdFloors[0].spots:
            tempSpot = tupleTransform(spot, origin3[0], rotate)
            thirdSpotsWorld.append(tempSpot)
            thirdFloors[0].worldSpots.append(np.append(tempSpot, getHeight(2)))

        allWorldSpots.append(thirdSpotsWorld)
        if printstuff == True:
            print("Spots level 3: " + str(thirdSpotsWorld))

        # randomly fill from 0 to 100%
        numPillars3 = len(thirdSpotsWorld)
        filled3 = np.zeros(numPillars3)
        # need at least half
        numChosen3 = random.sample(range(0, numPillars3//2), 1)[0]

        chosen3 = random.sample(range(0, numPillars3), numChosen3)
        for i in chosen3:
            filled3[i] = 1

        count = 0
        for i in thirdFloors:
            numSpots = len(i.spots)
            i.filled = filled3[count:numSpots+count]
            count += numSpots

        occupied3 = getOccupied([thirdSpotsWorld], [filled3])
        filledSpots.append(occupied3)
        allFilled.append(filled3)

        towerTiles.append(thirdFloors)
    if printstuff == True:
        # print("All occupied spots:")
        # print(filledSpots)
        print("Origins:")
        print(origins)
        print("All Filled:")
        print(allFilled)
        print("All spots:")
        print(allWorldSpots)
        print("Occupied Final:")
        getOccupied(allWorldSpots, allFilled)
        print()
        print("Self stuff:")
        print(towerTiles)
        print()
        for i in towerTiles:
            for j in i:
                # print("Level ", i, "Tile ", j)
                print(j.origin)
                print(j.rotation)
                print(j.worldSpots)
                print(j.filled)
                print()
    # print([len(i[0].worldSpots) for i in towerTiles])
    return allWorldSpots, allFilled, origins, towerTiles


# a = (5,10)
# b = (7, 11)
# origin = (20, 30)
# rotate = getRotateMat(0)
# print(rotate)
# x = tupleTransform(a, origin, rotate)
# print(x)
#
# print(checkCollide(a, origin))
# print(checkCollide(a,b))
# try:
#     for _ in range(10000):
#         build()
#
# except Exception as e:
#     print(e)

if __name__ == "__main__":
    numTowers = 300  # 317 towers saved out of 1000
    # fileName = "TestTowerSim_" + str(numTowers) + ".txt"
    fileName = "TestTowerSim_" + str(numTowers) + ".txt"
    file = open(fileName, 'wb')
    count = 0
    for _ in range(numTowers):
        goal = [-1]
        spots = -1
        while goal[0] == -1 or spots == -1:
            spots, filled, origins, tower = build()
            if spots == -1:
                continue  # won't be able to get goal
            goal = SawyerSim.getGoal(spots, filled)

        # check that it can be built in coppelia:
        collapsed = PhysicsConnect.simulate(tower)
        if collapsed: # try 1 more time
            # collapsed = PhysicsConnect.simulate(tower)
            # if collapsed:
            print("Discarded")
        # print(tower[0][0].worldSpots)
        # print(tower[-1][0].worldSpots)
        if not collapsed:
            # spots, filled, origins, tower = build() # remove if uncommenting
            pickle.dump([tower], file)
            print("Saved!")
            count += 1
    file.close()
    print(count)


# for a given tower that is missing the colour parameter, add it in
# def tempTransfer(tower):
#     ids = ['Tile01', 'Tile02', 'Tile03', 'Tile04', 'Tile05', 'Tile06', 'Tile07', 'Tile08', 'Tile09', 'Tile10', 'Tile11',
#            'Tile12', 'Tile13', 'Tile14', 'Tile15', 'Tile16', 'Tile17', 'Tile18']
#     newTower = []
#     for level in tower:
#         floor = []
#         for tile in level:
#             # get tile colours based on tile id
#             index = ids.index(tile.id)
#             colours = tilesData[index][3]
#             newTile = Tile(tile.id, tile.spots, rotation=tile.rotation,
#                            origin=tile.origin, outline=tile.outline, colours=colours)
#             newTile.worldSpots = np.copy(tile.worldSpots)
#             newTile.filled = np.copy(tile.filled)
#             floor.append(newTile)
#         newTower.append(floor)
#     return newTower
#
# # validation set - append to file instead of rewriting
# if __name__ == "__main__":
#     fileName = "Validation_25.txt"
#     goal = [-1]
#     spots = -1
#     count = 0
#     while goal[0] == -1 or spots == -1:
#         spots, filled, origins, tower = build()
#         if spots == -1:
#             continue  # won't be able to get goal
#         goal = SawyerSim.getGoal(spots, filled)
#
#     # check that it can be built in coppelia:
#
#     collapsed = PhysicsConnect.simulate(tower)
#     if len(tower) == 1:  # exclude level 1
#         collapsed = False
#     # if collapsed:  # try 1 more time
#     #     collapsed = PhysicsConnect.simulate(tower)
#     #     if collapsed:
#     #         print("Discarded")
#     if collapsed:
#         print("Discarded")
#     if not collapsed:
#
#         file = open(fileName, 'rb')
#         # iterate thru to end of pickle dump, add back with next obj
#         objs = []
#         while 1:
#             try:
#                 # if count == 15:
#                 #     break
#                 oldtower = pickle.load(file)
#                 # newTower = tempTransfer(tower)
#                 objs.append(oldtower)
#                 count += 1
#             except EOFError:
#                 break
#         file.close()
#         print("Speak now")
#         time.sleep(3)
#         print("Hold your peace")
#         file = open(fileName, 'wb')
#         for i in objs:
#             pickle.dump(i, file)
#         # spots, filled, origins, tower = build() # remove if uncommenting
#         # new tower
#         pickle.dump([tower], file)
#         print("Saved!")
#         file.close()
#     print(count)

