# Make sure to have the server side running in CoppeliaSim:
# in a child script of a CoppeliaSim scene, add following command
# to be executed just once, at simulation start:
#
# simRemoteApi.start(19999)
#
# then start simulation, and run this program.
#
# IMPORTANT: for each successful call to simxStart, there
# should be a corresponding call to simxFinish at the end!

import sim
from HighTowerSim import Tile
import SawyerSim
import sys
import numpy as np
import quaternion
import time
import pickle
import random


# pt1 has [x,y], pt2 has [x,y,z]. Ignore z
def getDist(pt1, pt2):
    dist = np.sqrt((pt1[0]-pt2[0])**2 + (pt1[1]-pt2[1])**2)
    return dist


# tower has up to 3 elements, each with a tile
# tiles have information on origin, rotation, spots (in world coords), and filled
# tiles can be placed at exact location, pillars must be placed at location + 4.5e-2
def simulate(tower):
    # print('Program started')
    sim.simxFinish(-1) # just in case, close all opened connections
    clientID=sim.simxStart('127.0.0.1', 19997, True, True, 5000, 5) # Connect to CoppeliaSim

    pillarHandles = []  # save all generated pillar handles
    pillarLocations = []  # save all locations the pillars should be at to double check (just x and y)
    parentTiles = []  # need to save the parent tile for pillars
    if clientID!=-1:
        # print ('Connected to remote API server')
        # Now try to retrieve data in a blocking fashion (i.e. a service call):
        res, objs=sim.simxGetObjects(clientID,sim.sim_handle_all,sim.simx_opmode_blocking)
        if res != sim.simx_return_ok:
            print ('Remote API function call returned with error code: ', res)
            sys.quit()
        # print(objs)
        # o = sim.simxGetObjectGroupData(clientID, sim.sim_appobj_object_type, 5, sim.simx_opmode_blocking)[3]
        # print(sim.simxGetObjectGroupData(clientID, sim.sim_appobj_object_type, 20, sim.simx_opmode_blocking))
        pillarHandle = sim.simxGetObjectHandle(clientID, "Pillar", sim.simx_opmode_blocking)[1]
        # print("Ori:")
        # print(o[3*pillarHandle:3*pillarHandle+3])
        # print(o)
        sim.simxStartSimulation(clientID, sim.simx_opmode_blocking)
        for level in tower:
            for tile in level:
                # for each tile, place at spot
                # get handle for tile
                handle = sim.simxGetObjectHandle(clientID, tile.id, sim.simx_opmode_blocking)[1]
                newTile = sim.simxCopyPasteObjects(clientID, [handle], sim.simx_opmode_blocking)[1][0]
                # position
                # print(tile.id)
                origin = [x/1000 for x in tile.origin] # convert from mm to m
                # print(origin)
                origin = [i*5 for i in origin] # multiply by 5 because of coppelia scaling
                origin[2] *= 0.9 # lower it a bit
                # set rotation first
                ori = sim.simxGetObjectQuaternion(clientID, newTile, -1, sim.simx_opmode_blocking)[1]
                oriQ = np.quaternion(ori[3], ori[0], ori[1], ori[2])
                rot = tile.rotation
                newQ = np.quaternion(np.cos(rot / 2), 0, 0, np.sin(rot / 2))
                quat = newQ * oriQ
                quat2 = [quat.x, quat.y, quat.z, quat.w]
                sim.simxSetObjectQuaternion(clientID, newTile, -1, quat2, sim.simx_opmode_blocking)
                sim.simxSetObjectPosition(clientID, newTile, -1, origin, sim.simx_opmode_blocking)
                # rotation
                # for each tile, fill pillars
                # time.sleep(1)
                sim_handle_parent = newTile

                # print("Pillars")
                # order pillars by distance from the origin of the piece
                dists = [getDist(i, origin) for i in tile.spots]
                sortedPillars = [i for _, i in sorted(zip(dists, tile.spots))]
                sortedFilled = [i for _, i in sorted(zip(dists, tile.filled))]

                for (binary, pillar) in zip(sortedFilled, sortedPillars):
                    if binary:
                        newPillar = sim.simxCopyPasteObjects(clientID, [pillarHandle], sim.simx_opmode_blocking)[1][0]
                        sim.simxSetObjectOrientation(clientID, pillarHandle, -1, [0, 0, 0], sim.simx_opmode_blocking)
                        pillarHandles.append(newPillar)
                        # make pillar, place in position
                        location = np.array([x for x in pillar])
                        location = np.concatenate((location/1000, [0.01]))*5
                        pillarLocations.append(location)
                        parentTiles.append(newTile)
                        # print(location)
                        sim.simxSetObjectPosition(clientID, newPillar, sim_handle_parent, location, sim.simx_opmode_blocking)
                    time.sleep(0.1)


                # sys.exit()

        # pos = sim.simxGetObjectPosition(clientID, 6, -1, sim.simx_opmode_oneshot)
        # print(pos)
        # data = sim.simxGetObjectGroupData(clientID, sim.sim_appobj_object_type, 20, sim.simx_opmode_blocking)
        # print(data)
        # handle = sim.simxGetObjectHandle(clientID, "Tile07", sim.simx_opmode_blocking)[1]
        # print(handle)
        #
        # sim.simxSetObjectPosition(clientID, handle, -1, [0, 0, 0.1], sim.simx_opmode_oneshot)
        # pos = sim.simxGetObjectPosition(clientID, handle, -1, sim.simx_opmode_oneshot)
        # print(pos)
        # # Now retrieve streaming data (i.e. in a non-blocking fashion):
        #
        # sim.simxGetIntegerParameter(clientID, sim.sim_intparam_mouse_x,sim.simx_opmode_streaming) # Initialize streaming
        #
        # # # Now send some data to CoppeliaSim in a non-blocking fashion:
        # sim.simxAddStatusbarMessage(clientID,'Hello CoppeliaSim!',sim.simx_opmode_oneshot)
        # #
        # # # Before closing the connection to CoppeliaSim, make sure that the last command sent out had time to arrive. You can guarantee this with (for example):
        # sim.simxGetPingTime(clientID)

    else:
        print('Failed connecting to remote API server')

    time.sleep(0.5)
    # print("Angles:")
    # angles = sim.simxGetObjectOrientation(clientID, handle, -1, sim.simx_opmode_blocking)[1]
    # print(np.rad2deg(angles))
    # figure out top tile's position, check if it's correct
    actualPos = sim.simxGetObjectPosition(clientID, newTile, -1, sim.simx_opmode_blocking) # should just be last one called
    initialPos = origin
    # can guarantee position will be less than 0.05 if tower collapses
    thresh = 0.55

    collapsed = 0
    # check distances of pillars
    count = 0
    # print(pillarHandles)
    # print(pillarLocations)
    # print("Distances:")
    # print(pos)

    pos = sim.simxGetObjectOrientation(clientID, newTile, -1, sim.simx_opmode_blocking)[1]
    if np.abs(pos[0]) + np.abs(pos[1]) > 0.1:
        collapsed = 1


    for i in pillarHandles:
        sim_handle_parent = parentTiles[count]
        pos = sim.simxGetObjectPosition(clientID, i, sim_handle_parent, sim.simx_opmode_blocking)
        # print(pos)
        # print(pillarLocations[count])
        dist = np.sqrt((pillarLocations[count][0] - pos[1][0])**2 + (pillarLocations[count][1] - pos[1][1])**2)
        # print(dist)
        # print()
        if dist > 0.1:
            collapsed = 1
        count += 1

    if actualPos[1][2]/initialPos[2] <= thresh and len(tower) != 1:
        collapsed = 1
        # print(actualPos[1][2]/initialPos[2])



    # if collapsed:
    #     print("Collapsed!")

    # i = "Tile05"
    # # print(sim.simxCheckCollision(clientID, 0, 1, sim.simx_opmode_streaming))
    # # for j in handles:
    # #     tempi = sim.simxGetObjectHandle(clientID, i, sim.simx_opmode_streaming)[1]
    # #     tempj = sim.simxGetObjectHandle(clientID, j, sim.simx_opmode_streaming)[1]
    #     # print(j, tempj)
    #     # print(sim.simxCheckCollision(clientID, tempi, tempj, sim.simx_opmode_streaming))
    # temp = sim.simxGetObjectGroupData(clientID, sim.sim_appobj_object_type, 20, sim.simx_opmode_blocking)
    # result = temp[0]
    # numHandle = temp[1]
    # stringHandle = temp[4]
    # sensorHandle = sim.simxGetObjectHandle(clientID, 'Ps', sim.simx_opmode_blocking)
    # print(sensorHandle)
    # # temp = sim.simxGetObjectGroupData(clientID, sim.sim_appobj_object_type, 13, sim.simx_opmode_blocking)
    # temp = sim.simxReadProximitySensor(clientID, sensorHandle[1], sim.simx_opmode_blocking)
    # ind = numHandle.index(temp[3])
    # print(ind)
    # print(stringHandle[ind])
    # print(temp)

    # Now close the connection to CoppeliaSim:
    sim.simxStopSimulation(clientID, sim.simx_opmode_blocking)
    sim.simxFinish(clientID)
    # print('Program ended')
    return collapsed

if __name__ == "__main__":
    # tower is the only one we really care about
    # others are just to check that tower is viable
    numTowers = 1
    fileName = "MinTowerSim_" + str(numTowers) + ".txt"
    f = open(fileName, 'rb')
    temp = pickle.load(f)
    tower = temp[0]
    f.close()
    simulate(tower)