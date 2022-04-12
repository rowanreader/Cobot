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
import HighTowerSim
import SawyerSim
import sys
import numpy as np
import quaternion
import time

# tower is the only one we really care about
# others are just to check that tower is viable

goal = [-1]
spots = -1
while goal[0] == -1 or spots == -1:
    spots, filled, origins, tower = HighTowerSim.build()

    if spots == -1:
        continue  # won't be able to get goal
    goal = SawyerSim.getGoal(spots, filled)


# tower has up to 3 elements, each with a tile
# tiles have information on origin, rotation, spots (in world coords), and filled


# tiles can be placed at exact location, pillars must be placed at location + 4.5e-2

print ('Program started')
sim.simxFinish(-1) # just in case, close all opened connections
clientID=sim.simxStart('127.0.0.1', 19997, True, True, 5000, 5) # Connect to CoppeliaSim
if clientID!=-1:
    print ('Connected to remote API server')

    # Now try to retrieve data in a blocking fashion (i.e. a service call):
    res, objs=sim.simxGetObjects(clientID,sim.sim_handle_all,sim.simx_opmode_blocking)
    if res == sim.simx_return_ok:
        print ('Number of objects in the scene: ',len(objs))
        print(objs)
    else:
        print ('Remote API function call returned with error code: ', res)
        sys.quit()
    print(tower)
    pillarHandle = sim.simxGetObjectHandle(clientID, "Pillar", sim.simx_opmode_blocking)[1]
    sim.simxStartSimulation(clientID, sim.simx_opmode_blocking)
    for level in tower:
        for tile in level:
            # for each tile, place at spot
            # get handle for tile
            # tile.id = "Tile12"

            handle = sim.simxGetObjectHandle(clientID, tile.id, sim.simx_opmode_blocking)[1]
            # position
            print(tile.id)
            origin = (x/1000 for x in tile.origin) # convert from mm to m
            sim.simxSetObjectPosition(clientID, handle, -1, origin, sim.simx_opmode_blocking)
            pos = sim.simxGetObjectPosition(clientID, handle, -1, sim.simx_opmode_blocking)
            # print(pos)
            # rotation
            ori = sim.simxGetObjectQuaternion(clientID, handle, -1, sim.simx_opmode_blocking)[1]

            oriQ = np.quaternion(ori[3], ori[0], ori[1], ori[2])
            rot = np.deg2rad(45)
            # print(tile.rotation)
            newQ = np.quaternion(np.cos(rot/2), 0, 0, np.sin(rot/2))
            # print(tile.id)
            quat = newQ*oriQ
            quat2 = [quat.x, quat.y, quat.z, quat.w]
            sim.simxSetObjectQuaternion(clientID, handle, -1, quat2, sim.simx_opmode_blocking)
            # for each tile, fill pillars

            for (binary, pillar) in zip(tile.filled, tile.worldSpots):
                if binary:
                    print(pillarHandle)
                    newPillar = sim.simxCopyPasteObjects(clientID, [pillarHandle], sim.simx_opmode_blocking)[1][0]
                    # make pillar (?) place in position
                    location = pillar/1000
                    sim.simxSetObjectPosition(clientID, newPillar, -1, location, sim.simx_opmode_blocking)
                time.sleep(0.5)
            time.sleep(0.5)

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
    #

else:
    print ('Failed connecting to remote API server')

# # Now close the connection to CoppeliaSim:
time.sleep(2)
sim.simxStopSimulation(clientID, sim.simx_opmode_blocking)
sim.simxFinish(clientID)
print ('Program ended')