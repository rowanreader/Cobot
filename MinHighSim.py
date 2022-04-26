# the same as TowerSim but with different outputs:
# [((x, y, z, theta) (x, y, z, filled) * 5) *3] * 3
# origin of floor + angle, all points on that floor in global coords, 1 array/level
import random
import numpy as np
import sys
import pickle
import cdd as pcdd
from scipy.spatial import ConvexHull
import matplotlib.pyplot as plt
import SawyerSim

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
    return origin

    # check error size, once less than threshold (or number of iterations is too large), stop

    # check that there is one x coord larger and 1 x coord smaller. similar for y. if not, print error


class Tile:
    # spots is array of tupples, contatining distance from com of tile in x and y (mm)
    # rotation is angle of rotation the tile is in (radians)
    def __init__(self, Id, spots, rotation=0, origin=(0, 0, 0), outline=[(0,0)]):
        self.id = Id
        self.spots = spots
        self.rotation = rotation
        self.origin = origin
        self.worldSpots = [] # based on rotation and level
        self.filled = [] # binary array of filled or not filled, corresponding to worldSpots
        self.outline = outline

# list of all tiles and their spots relative to their origin (mm)
# only ints allowed
# Dagger 01
tile01 = Tile("Tile01", [(-3, 47), (20, 10), (-1, -42)], outline=[(20, 90), (35, 8), (6, -90), (-26, -28), (0, 30), (-23, 30)])  # (R, Y, R)
# Majora's Mask
tile02 = Tile("Tile02", [(27, 10), (-5, -26), (-22, 17)], outline=[(0, 68), (62, 25), (43, -50), (-40, -50), (-60, 24)])  # (K, K, R)
# Shard
tile03 = Tile("Tile03", [(-61, -15), (-5, 13), (16, -18), (55, -3)], outline=[(0, 25), (94, -15), (13, -40), (-16, -16), (-60, -22), (-104, -5)])  # (R, Y, R, W)
# Square Spiral
tile04 = Tile("Tile04", [(-34, 45), (-11, -10), (13, 10), (35, -45)], outline=[(-45, 60), (57, 43), (45, -65), (-55, -43)])  # (Y, K, K, Y)
# Toucan
tile05 = Tile("Tile05", [(-34, -13), (-6, -22), (23, 25), (34, -11)], outline=[(85, 17), (10, 60), (-25, 5), (-57, 15), (-55, 43), (-78, -14), (0, -54), (70, -30)])  # (R, K, Y, Y)
# Crescent
tile06 = Tile("Tile06", [(-31, 19), (2, 33), (34, 16)], outline=[(43, -57), (50, 10), (0, 35), (-52, 6), (-40, -60), (-35, -15), (0, 3), (35, 15)])  # (Y, R, W)
# Half Ax Head
tile07 = Tile("Tile07", [(-26, -25), (2, 34), (11, -20)], outline=[(0, 60), (-25, 31), (-27, -15), (-68, -14), (0, -45), (67, -12), (28, 38)])  # (Y, Y, Y)
# Ax
tile08 = Tile("Tile08", [(-75, 26), (-40, -17), (-35, 38), (83, -23)], outline=[(-60, -27), (135, 15), (-7, -2), (10, 40), (25, 48), (-43, 62), (-72, 14)])  # (Y, K, K, Y)
# Triskelion
tile09 = Tile("Tile09", [(-53, -47), (-1, 61), (55, -47)], outline=[(-15, 85), (-70, -55), (-81, 33)])  # (K, Y, W)
# Orchid
tile10 = Tile("Tile10", [(-33, -34), (-13, 38), (-3, -35), (31, -19), (39, 39)], outline=[(60, 65), (40, 11), (53, -14), (-17, -53), (-98, 12), (-4, -3), (-28, 17), (-41, 65), (10, 35)])  # (Y, K, Y, B, K)
# Red Square
tile11 = Tile("Tile11", [(-24, 28), (-13, -38), (28, -39), (32, 31)], outline=[(19, 70), (83, 19), (35, -55), (10, -43), (-51, -46), (-34, 0), (-52, 35), (62, 15)])  # (R, R, R, R)
# Massive
tile12 = Tile("Tile12", [(-65, 0), (-39, 0), (1, -39), (1, 39), (41, 0)], outline=[(0, 79), (45, 30), (110, 0), (53, -30), (0, -80), (-38, -33), (-110, 0), (-43, 33)])# (W, Y, R, B, Y)
# Square
tile13 = Tile("Tile13", [(-11, -12), (-11, 11), (11, 11), (45, 39), (42, -42)], outline=[(65, 67), (40, 0), (70, -65), (0, 40), (-64, -67), (-40, 0), (-67, 64), (0, 39)])  # (R, W, R, W, Y)
# Ax Head
tile14 = Tile("Tile14", [(-15, 37), (0, 0), (14, 38)], outline=[(42, 32), (14, 0), (41, -35), (0, -55), (-44, -34), (-16, 0), (-41, 35), (0, 54)])  # (Y, R, B)
# Flame
tile15 = Tile("Tile15", [(-17, 10), (-1, 22), (20, 10)], outline = [(40, 11), (12, -45), (0, -2), (-10, -20), (9, -68), (-36, 13), (0, 41)])  # (Y, R, Y)
# Shuriken
tile16 = Tile("Tile16", [(-24, -11), (0, 0), (1, 26), (22, -16)], outline=[(37, 22), (30, -35), (0, -45), (-44, -10), (-40, 20), (15, 41)])  # (Y, K, R, Y)
# Stairs
tile17 = Tile("Tile17", [(-60, -8), (-22, 28), (16, -3), (60, -6)], outline=[(45, 40), (75, 18), (80, -53), (0, 10), (-80, -53), (-72, 18), (-43, 30)])  # (Y, Y, R, Y)
# Spaceship
tile18 = Tile("Tile18", [(-26, 32), (0, 9), (26, 32)], outline=[(55, 46), (13, -26), (7, -87), (-10, -30), (-53, 43), (0, 30)])  # (R, K, Y)

tile19 = Tile("Tile19", [(-53, -47), (-1, 61), (55, -47)], outline=[(-15, 85), (-70, -55), (-81, 33)])  # (K, Y, W)
tile20 = Tile("Tile20", [(-53, -47), (-1, 61), (55, -47)], outline=[(-15, 85), (-70, -55), (-81, 33)])  # (K, Y, W)

# tiles = [tile09, tile19, tile20]
tiles = [tile01, tile02, tile03, tile04, tile05, tile06, tile07, tile08, tile09, tile10, tile11, tile12, tile13, tile14,
         tile15, tile16, tile17, tile18]

tiles1 = [tile10, tile12, tile05, tile11, tile08] # tile09, tile06
tiles2 = [tile07, tile01, tile03, tile18, tile13, tile04]
tiles3 = [tile14, tile17, tile02, tile16, tile15]

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

def plotHull(hull1, hull2, intersect):
    plt.plot(hull1[:,0], hull1[:,1], color="blue")
    plt.plot([x[0] for x in hull2], [y[1] for y in hull2], color="green")
    plt.plot([float(x) for x in intersect[:, 0]], [float(x) for x in intersect[:, 1]], color="red")

    plt.show()


# builds random tower
# returns occupied spots and origin of tiles
# can make plane based on that

def build():
    taken = [] # holds already used tiles
    allWorldSpots = []
    towerTiles = []
    allFilled = []
    # randomly choose 3 tiles and place in fixed origin spots
    temp1 = len(tiles1)
    firstFloorId = random.sample(range(0, temp1), 1)  # corresponds to index in tiles, so number 0 = tile01
    # firstFloorId = [0]

    taken += firstFloorId
    # firstFloorId = [2, 11, 15]
    firstFloors = [tiles1[firstFloorId[0]]]
    if printstuff == True:
        print(firstFloorId)
    origin1 = [[500, 100], [600, 300], [700, 0]]  # mm from origin of world coord
    count = 0
    for i in firstFloors:
        i.origin = np.append(origin1[count], getHeight(0))
        count += 1

    origins = [origin1]

    # randomly choose level (1, 2, 3)
    levels = random.sample(range(1, 4), 1)[0]  # either 1 2 or 3
    levels = 3
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
        numChosen1 = random.sample(range(0, numPillars1), 1)[0]

    elif levels == 2:
        numChosen1 = random.sample(range(numPillars1 // 2 - 1, numPillars1), 1)[0]  # limit the number of pillars to be between half and all

    elif levels == 3:
        numChosen1 = random.sample(range(int(numPillars1 * 0.8) - 1, numPillars1), 1)[0]  # limit the number of pillars to be between half and all

    numChosen1 = numPillars1  # select all
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
        temp2 = len(tiles2)
        secondFloorId = random.sample(range(0, temp2), 1)

        # secondFloorId = [1]
        taken += secondFloorId
        if printstuff == True:
            print(secondFloorId)
        secondFloors = []
        for i in secondFloorId:
            secondFloors.append(tiles2[i])

        origin2 = getClusters(firstSpotsWorld, filled1, 1)
        # origin2 = origin1
        count = 0
        for i in secondFloors:
            i.origin = np.append(origin2[count], getHeight(1))
            count += 1
        origins.append(origin2)
        if printstuff == True:
            print("Level 2 origins: " + str(origin2))
        # choose origin in center of triangle of pillars??? Given pillars and number of 'clusters' find centroids
        # orient so no collisions
        secondSpotsWorld = []
        # place 1st floor of 2nd level in neutral
        for spot in secondFloors[0].spots:
            tempSpot = tupleTransform(spot, origin2[0], rotate)
            secondSpotsWorld.append(tempSpot)
            secondFloors[0].worldSpots.append(np.append(tempSpot, getHeight(1)))

        # given a tile, origin (centriod) and POS, find rotation of max overlap using convexhull
        angle = getRotation(secondFloors[0], origin2, firstSpotsWorld)
        secondFloors[0].rotation = angle # this is actually the most recent angle from 1st floor

        allWorldSpots.append(secondSpotsWorld)
        if printstuff == True:
            print("Spots level 2: " + str(secondSpotsWorld))

        # fill in spots
        numPillars2 = len(secondSpotsWorld)
        filled2 = np.zeros(numPillars2)

        if levels == 2:  # 0 to 100%
            numChosen2 = random.sample(range(0, numPillars2), 1)[0]  # limit the number of pillars to be between half and all


        elif levels == 3:  # 60% to 100%
            numChosen2 = random.sample(range(int(numPillars2 * 0.6) - 1, numPillars2), 1)[0]  # limit the number of pillars to be between half and all

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
        temp3 = len(tiles3)
        thirdFloorId = random.sample(range(0, temp3), 1)
        # thirdFloorId = [2]
        if printstuff == True:
            print(thirdFloorId)
        thirdFloors = []
        for i in thirdFloorId:
            thirdFloors.append(tiles3[i])

        origin3 = getClusters(secondSpotsWorld, filled2, 1)
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
        rotate = np.identity(2)
        # place 1st floor of 3rd level in neutral
        for spot in thirdFloors[0].spots:
            tempSpot = tupleTransform(spot, origin3[0], rotate)
            thirdSpotsWorld.append(tempSpot)
            thirdFloors[0].worldSpots.append(np.append(tempSpot, getHeight(2)))

        angle = getRotation(thirdFloors[0], origin3, secondSpotsWorld)
        thirdFloors[0].rotation = angle

        allWorldSpots.append(thirdSpotsWorld)
        if printstuff == True:
            print("Spots level 3: " + str(thirdSpotsWorld))

        # randomly fill from 0 to 100%
        numPillars3 = len(thirdSpotsWorld)
        filled3 = np.zeros(numPillars3)
        # need at least 1 free
        numChosen3 = random.sample(range(0, numPillars3), 1)[0]

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
    numTowers = 1
    fileName = "MinTowerSim_" + str(numTowers) + ".txt"
    file = open(fileName, 'wb')
    for _ in range(numTowers):
    #     goal = [-1]
    #     spots = -1
    #     while goal[0] == -1 or spots == -1:
    #         spots, filled, origins, tower = build()
    #
    #         if spots == -1:
    #             continue  # won't be able to get goal
    #         goal = SawyerSim.getGoal(spots, filled)
        spots, filled, origins, tower = build() # remove if uncommenting
        pickle.dump([tower], file)
    file.close()
