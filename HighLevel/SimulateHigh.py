import pychrono.core as chrono
import pychrono.irrlicht as chronoirr
import numpy as np
import time
chrono.SetChronoDataPath('/home/jacqueline/anaconda3/pkgs/pychrono-7.0.0-py38_0/share/chrono/data/')


class ContactReporter(chrono.ReportContactCallback):
    def __init__(self, box):
        self.m_box = box
        self.collided = False
        super().__init__()

    def OnReportContact(self, pA, pB, plane_coord, distance, eff_radius, cforce, ctorque, modA, modB):
        if -0.05 < distance < 0.05:
            # only print once
            if not self.collided:
                print(distance)
                self.collided = True
            return True

        return False


class ContactMaterial(chrono.AddContactCallback):
    def __init__(self):
        super().__init__()
    def OnAddContact(self, contactinfo, material):
        # Downcast to appropriate composite material type
        mat = chrono.CastToChMaterialCompositeNSC(material)

        friction = 0.3
        mat.static_friction = friction
        mat.sliding_friction = friction


class Pillar:
    # just take in system and coords
    def __init__(self, system, pos):
        # hard coded stuff - divide by 100
        divide = 10
        self.height = 45/divide
        self.radius = 6/divide
        wood = chrono.ChMaterialSurfaceNSC()
        # radius, height half length, density, visualize, collide, material
        # hard code
        body = chrono.ChBodyEasyCylinder(self.radius, self.height / 2, 10, True, True, wood)
        body.SetBodyFixed(False)
        body.SetPos(chrono.ChVectorD(pos[0], pos[1], pos[2]))

        system.Add(body)


class Tile:
    # takes in system, if floor is start, floorDef, ID, mass (g), coordinates of origin of floor in global, angle of x axis, spots in local
    def __init__(self, system, start, tile, origin, angle):
        divide = 1
        self.floorDef = tile[0] # list of tuples with floor parts
        self.mass = tile[1]
        self.x = origin[0]/divide
        self.y = origin[1]/divide
        self.z = origin[2]/divide
        self.rotation = tile[2]
        # self.inertia = tile[3]
        # self.angle = np.deg2rad(angle)
        # self.com = [origin[0], origin[1], origin[2]] # com is initially the origin
        material = chrono.ChMaterialSurfaceNSC()
        material.SetFriction(0.1)
        # make and add to system


        mrigidBody = chrono.ChBody()
        thickness = 0.5  # tile thickness
        mrigidBody.SetBodyFixed(start)  # if start is true, make fixed, is base of tower()

        mrigidBody.SetMass(self.mass)
        mrigidBody.SetDensity(100.0)
        for part in self.floorDef: # iterate through components of tile to build shape
            # mrigidBody = chrono.ChBody()
            shape = part[0]

            # send in angle and origin of rotation

            if shape == 0:  # rectangle
                rotation = self.rotate(part[6])  # get rotation matrix for entire tile
                translation = chrono.ChVectorD(part[3], part[4], part[5])


                vshape = chrono.ChBoxShape()
                d = self.mass/(part[1] * thickness * part[2])
                temp = chrono.ChBodyEasyBox(part[1], thickness, part[2], d)
                inertia = temp.GetInertia()
                # quat = chrono.ChVectorD(self.inertia[0], self.inertia[1], self.inertia[2])
                # qrot = chrono.ChMatrix33D(quat)

                # vshape.Rot = rotation
                # half dimensions
                # mrigidBody.GetCollisionModel().AddBox(material, part[1]/2,  0.5/2, part[2]/2,
                #                                       chrono.ChVectorD(0, 0, 0), rotation)
                mrigidBody.GetCollisionModel().AddBox(material, part[1] / 2, thickness / 2, part[2] / 2,
                                                      translation, rotation)


                mrigidBody.SetCollide(True)
                vshape.GetBoxGeometry().SetLengths(chrono.ChVectorD(part[1], thickness, part[2]))

                # # box center
                vshape.GetBoxGeometry().Pos = translation
                vshape.GetBoxGeometry().Rot = rotation
                mrigidBody.AddAsset(vshape)

                # mrigidBody.SetInertia(inertia)
                # mrigidBody.SetShowCollisionMesh(True)
                # vshape.GetBoxGeometry().Pos = chrono.ChVectorD(part[3], part[4], part[5])


            elif shape == 1:  # triangle based, can assume only 1 shape
                points = part[1] # array of points

                vectors = []
                translation = chrono.ChVectorD(part[2], part[3], part[4])
                for point in points:
                    vectors.append(chrono.ChVectorD(point[0], point[1], point[2]) + translation)
                    vectors.append(chrono.ChVectorD(point[0], point[1]-thickness, point[2]) + translation)
                    print(point)

                # v1 = chrono.ChVectorD(top[0], top[1], top[2]) + translation
                # v2 = chrono.ChVectorD(left[0], left[1], left[2]) + translation
                # v3 = chrono.ChVectorD(right[0], right[1], right[2]) + translation
                #
                # v4 = chrono.ChVectorD(btop[0], btop[1], btop[2]) + translation
                # v5 = chrono.ChVectorD(bleft[0], bleft[1], bleft[2]) + translation
                # v6 = chrono.ChVectorD(bright[0], bright[1], bright[2]) + translation
                mpoints = chrono.vector_ChVectorD(vectors)

                mrigidBody = chrono.ChBodyEasyConvexHullAuxRef(mpoints, 1000, True, True, chrono.ChMaterialSurfaceNSC())

                # mrigidBody.Move(chrono.ChVectorD(2, 0.3, 0))

            elif shape == 2:  # circle = really short cylinder, assume only 1 shape
                # send in origin, origin adjusted for height, radius
                center = chrono.ChVectorD(part[2], part[3], part[4])

                radius = part[1]
                # radius, height, density, visualize, collide
                mrigidBody = chrono.ChBodyEasyCylinder(radius, thickness, 10, True, True, material)
                mrigidBody.SetPos(center)


            # print(translation)
            # print(rotation)
            # print(0)
            # mrigidBody.GetCollisionModel().ClearModel()
            # mrigidBody.AddAsset(vshape)

            mrigidBody.GetCollisionModel().BuildModel()

        system.Add(mrigidBody)
        # tileRotate = self.rotate(self.rotation)
        # mrigidBody.SetRot(tileRotate)

    # returns 3x3 rotation matrix of place rotated about y (up) axis by angle
    def rotate(self, angle):
        angle = np.deg2rad(angle)
        matrix = chrono.ChMatrix33D(angle, chrono.ChVectorD(0, 1, 0))
        return matrix



# adds the fixed ground that registers collisions
def addGround(system, origin):
    material = chrono.ChMaterialSurfaceNSC() # default friction 0.6
    # material.SetFriction(0.1)
    # xsize, ysize, zsize, density, visualize, collide, material
    floorBody = chrono.ChBodyEasyBox(100/2, 1/2, 100/2, 100, True, True, material)
    floorBody.SetPos(chrono.ChVectorD(origin[0], origin[1], origin[2]))
    floorBody.SetBodyFixed(True) # fixed
    # floorBody.GetCollisionModel().SetEnvelope(0.01)
    system.Add(floorBody)

    texture = chrono.ChTexture()
    texture.SetTextureFilename(chrono.GetChronoDataFile("textures/concrete.jpg"))
    floorBody.AddAsset(texture)

    creporter = ContactReporter(floorBody)
    cmaterial = ContactMaterial()
    return creporter, cmaterial

# for visualization
def visualize(system, creporter):
    application = chronoirr.ChIrrApp(system, 'PyChrono example: Collisions between objects',chronoirr.dimension2du(1024, 768))
    application.AddTypicalSky()
    application.AddTypicalLogo(chrono.GetChronoDataFile('logo_pychrono_alpha.png'))
    application.AddTypicalCamera(chronoirr.vector3df(0, 14, -60))
    application.AddTypicalLights()

    application.AssetBindAll()
    application.AssetUpdateAll()

    # application.SetTimestep(0.02)
    application.SetTryRealtime(True)
    while (application.GetDevice().run()):

        application.BeginScene()
        application.DrawAll()
        application.DoStep()
        application.EndScene()
        system.GetContactContainer().ReportAllContacts(creporter)




if __name__ == "__main__":
    # +x = right, +y = up, +z = into page
    origin = [0, -20, 0]

    # initialize system
    system = chrono.ChSystemNSC()
    system.Set_G_acc(chrono.ChVectorD(0, -10, 0))
    tileHeight = -6.0
    start = False


    # initialize tiles
    # IGNORING ROTATION FOR NOW

    # system, start (0th floor), tiledef, coordinates of origin of floor in global, angle of x axis
    # for tileDef has shape and mass (g), overall rotation, and inertia
    # 1st val of shape represents rect (0), triangle (1), and circle (2).
    # Remaining vals are x & z half-lengths (in cm), position, and angle about y (up axis)

    dropHeight = 5
    tile01 = ([(0, 6, 18, 0, tileHeight, 0, 0)],  20, 45, (145912, 162000, 16312.5))
    tile05 = ([(0, 7.5, 3.25, 0, tileHeight, 0, 0), (0, 3.5, 3, 4, tileHeight, 3, 0)], 30, 60, (145.912, 162.000, 16.3125))

    tile07 = ([(0, 5.5, 7, 0, tileHeight, 0, 0), (0, 13.5, 3, 0, tileHeight, -4, 0)], 10, 60, (150000, 150000, 150000))

    tile08 = ([(0, 6, 5, 0, tileHeight, 0, 0),(0, 7, 2, 6.5, tileHeight, -1.5, 0)], 30)

    tile10 = ([(0, 7.5, 2.5, 0, tileHeight, 0, 0),
              (0, 2, 3.25, -0.5, tileHeight, 3, 0),
              (0, 2.25, 3.5, 2.5, tileHeight, 3, 0)],
              20)

    # Neither box nor circle, defined by points
    hHeight = 15.5
    hWidth = 11.0
    center = (0.0, tileHeight, 0.0)
    top = np.array(center)  # need to avoid aliasing
    left = np.array(center)
    right = np.array(center)
    bottom = np.array(center)
    top[2] += hHeight  # top vertex goes into page
    left[0] -= hWidth
    right[0] += hWidth
    bottom[2] -= hHeight
    # takes in points, origin (translation) and rotation (for entire piece - not obsolete)
    tile12 = ([(1, [top, left, right, bottom],  -12.0, tileHeight, 0.0)], 10, 40)

    tile13 = ([(0, 4, 4, 0, tileHeight, 0, 0), (0, 2, 3, -2, tileHeight, 2, -45), (0, 2, 3, 2, tileHeight, 2, 45), (0, 2, 3, -2, tileHeight, -2, 45), (0, 2, 3, 2, tileHeight, -2, -45)], 20, 40, (145912, 162000, 16312.5))

    tile14 = ([(0, 4, 2.25, -2, tileHeight, 0, 0), (0, 4, 2.25, 2, tileHeight, 0, 0)], 10) # if keeping as box, just have 1 item

    # for cylinder, send in radius, and origin, height is assumed. No rotation needed
    tile16 = ([(2, 4.5, 0.0, tileHeight, 0.0)], 20,0)

    Tile(system, start, tile01, [origin[0], origin[1], origin[2]], 10)

    creporter, cmaterial = addGround(system, origin)
    system.GetContactContainer().RegisterAddContactCallback(cmaterial)

    visualize(system, creporter)