import time
import pybullet as pb
import pybullet_data
import os

client = pb.connect(pb.GUI) # switch to pb.DIRECT for non GUI

pb.setAdditionalSearchPath(pybullet_data.getDataPath())

pb.setGravity(0,0,-9.8) # z is up and down here

# shift = [0, -0.02, 0]
shift1 = [0, 1, 0]
shift2 = [0, 0, 2]

meshScale = [0.1, 0.1, 0.1]

# visualShapeId = p.createVisualShapeArray(shapeTypes=[p.GEOM_MESH, p.GEOM_BOX],
#                                          halfExtents=[[0, 0, 0], [0.1, 0.1, 0.1]],
#                                          fileNames=["duck.obj", ""],
#                                          visualFramePositions=[
#                                              shift1,
#                                              shift2,
#                                          ],
#                                          meshScales=[meshScale, meshScale])
# collisionShapeId = p.createCollisionShapeArray(shapeTypes=[p.GEOM_MESH, p.GEOM_BOX],
#                                                halfExtents=[[0, 0, 0], [0.1, 0.1, 0.1]],
#                                                fileNames=["duck_vhacd.obj", ""],
#                                                collisionFramePositions=[
#                                                    shift1,
#                                                    shift2,
#                                                ],
#                                                meshScales=[meshScale, meshScale])

name_in = "ObjectFiles/Tile10.obj"
name_out = "ObjectFiles/Tile10_vhacd.obj"
name_log = "log.txt"
pb.vhacd(name_in, name_out, name_log)

print("Done")
# make tiles
visualShapeId = pb.createVisualShapeArray(shapeTypes=[pb.GEOM_MESH],
                                         halfExtents=[[10, 10, 0]],
                                         fileNames=["ObjectFiles/Tile10.obj"],
                                         visualFramePositions=[
                                             shift2,
                                         ],
                                         meshScales=[meshScale, meshScale])


collisionShapeId = pb.createCollisionShapeArray(shapeTypes=[pb.GEOM_MESH],
                                               halfExtents=[[0, 0, 0]],
                                               flags=[pb.GEOM_FORCE_CONCAVE_TRIMESH],
                                               fileNames=["ObjectFiles/Tile10.obj"],
                                               collisionFramePositions=[
                                                   shift2,
                                               ],
                                               meshScales=[meshScale, meshScale])


# make pillar
visualCy = pb.createVisualShape(shapeType=pb.GEOM_CYLINDER,
                                     radius=0.2)

collisionCy = pb.createCollisionShape(shapeType=pb.GEOM_CYLINDER,
                                           radius=0.2)

mb = pb.createMultiBody(baseMass=1,
                           baseInertialFramePosition=[0, 0, 0],
                           baseCollisionShapeIndex=collisionCy,
                           baseVisualShapeIndex=visualCy,
                           basePosition=[0, 0, 1], # middle of object/cylinder
                           useMaximalCoordinates=False)


# ground
pb.loadURDF("plane100.urdf", useMaximalCoordinates=True)
rangex = 1
rangey = 1
for i in range(rangex):
  for j in range(rangey):
    mb = pb.createMultiBody(baseMass=1,
                           baseInertialFramePosition=[0, 0, 0],
                           baseCollisionShapeIndex=collisionShapeId,
                           baseVisualShapeIndex=visualShapeId,
                           basePosition=[((-rangex / 2) + i * 2),
                                         (-rangey / 2 + j * 2),
                                         2],
                           useMaximalCoordinates=False)

    # pb.changeVisualShape(mb, -1, rgbaColor=[1, 1, 1, 1])
pb.configureDebugVisualizer(pb.COV_ENABLE_RENDERING, 1)
# pb.setRealTimeSimulation(1)

# run simulation
pb.stepSimulation()
time.sleep(1)
for i in range(10000):

    pb.stepSimulation()
    time.sleep(1/140)


pb.disconnect()