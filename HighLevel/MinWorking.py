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
            # print(distance)
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
    def __init__(self, system, pos):
        wood = chrono.ChMaterialSurfaceNSC()
        # radius, height half length, density, visualize, collide, material
        # hard code
        body = chrono.ChBodyEasyCylinder(.6, 4.5/2, 10, True, True, wood)
        body.SetBodyFixed(False)
        body.SetPos(chrono.ChVectorD(pos[0], pos[1], pos[2]))

        system.Add(body)


class Tile:
    def __init__(self, system, floorDef):

        material = chrono.ChMaterialSurfaceNSC()
        material.SetRestitution(0.4)
        material.SetFriction(0.9)
        # material.SetCompliance(1)
        # material.SetDampingF(0)
        # make and add to system
        mrigidBody = chrono.ChBody()
        mrigidBody.SetBodyFixed(False)
        mrigidBody.SetCollide(True)
        mrigidBody.GetCollisionModel().ClearModel()
        for part in floorDef: # iterate through components of tile to build shape
            shape = part[0]
            # print(shape)
            if shape == 0:  # doens't work

                mrigidBody.SetMass(1.0)
                mrigidBody.SetDensity(1000.0)
                # quat = chrono.ChQuaternionD(1, 0 ,0, 0)
                quat = chrono.ChVectorD(145912, 162000, 16312.5)
                rot = chrono.ChMatrix33D(quat)

                mrigidBody.SetInertia(rot)
                vshape = chrono.ChBoxShape()
                # half dimensions
                # mrigidBody.GetCollisionModel().AddBox(material, part[1]/2,  part[1]/2, part[1]/2, chrono.ChVectorD(part[3], part[4], part[5]))
                mrigidBody.GetCollisionModel().AddBox(material, part[1]/2,  0.5/2, part[2]/2, chrono.ChVectorD(part[3], part[4], part[5]))
                vshape.GetBoxGeometry().SetLengths(chrono.ChVectorD(part[1], 0.5, part[2]))
                # vshape.GetBoxGeometry().SetLengths(chrono.ChVectorD(part[1], part[1], part[1]))
                # # box center
                vshape.GetBoxGeometry().Pos = chrono.ChVectorD(part[3], part[4], part[5])
                mrigidBody.AddAsset(vshape)

            elif shape == 1: # works
                mrigidBody = chrono.ChBodyEasyBox(part[1],  0.5, part[2], 100, True, True, material)
                mrigidBody.SetPos(chrono.ChVectorD(part[3], part[4], part[5]))
                print(material.GetRestitution())

            mrigidBody.GetCollisionModel().BuildModel()
        # mrigidBody.SetShowCollisionMesh(True)
        print("Values!!!")
        print(mrigidBody.GetMass())
        print(mrigidBody.GetForceList())
        print(mrigidBody.GetDensity())
        print(mrigidBody.GetInertia().Get_A_quaternion())
        print(mrigidBody.GetInertia().Get_A_Xaxis())
        print(mrigidBody.GetInertia().Get_A_Yaxis())
        print(mrigidBody.GetInertia().Get_A_Zaxis())
        print(mrigidBody.GetAssets())
        print(mrigidBody.GetCoord())
        # print(mrigidBody.Get)

        system.Add(mrigidBody)


# adds the fixed ground that registers collisions
def addGround(system, origin):
    material = chrono.ChMaterialSurfaceNSC() # default friction 0.6
    material.SetFriction(0.9)
    material.SetRestitution(0.9)
    # material.SetCompliance(1)
    # xsize, ysize, zsize, density, visualize, collide, material
    floorBody = chrono.ChBodyEasyBox(50, 0.5, 50, 100, True, True, material)
    floorBody.SetPos(chrono.ChVectorD(origin[0], origin[1], origin[2]))

    # floorBody = chrono.ChBody()
    # floorBody.SetCollide(True)
    # floorBody.SetDensity(100)
     # fixed
    # # half size
    # floorBody.GetCollisionModel().AddBox(material, 50 / 2, 0.5 / 2, 50 / 2,
    #                             chrono.ChVectorD(origin[0], origin[1], origin[2]))
    # floorBody.GetCollisionModel().SetEnvelope(0.01)

    floorBody.SetBodyFixed(True)
    system.Add(floorBody)

    texture = chrono.ChTexture()
    texture.SetTextureFilename(chrono.GetChronoDataFile("textures/concrete.jpg"))
    floorBody.AddAsset(texture)

    # vshape = chrono.ChBoxShape()
    # full length
    # vshape.GetBoxGeometry().SetLengths(chrono.ChVectorD(50, 0.5, 50))
    # # # box center
    # vshape.GetBoxGeometry().Pos = chrono.ChVectorD(origin[0], origin[1], origin[2])
    # floorBody.AddAsset(vshape)

    creporter = ContactReporter(floorBody)
    cmaterial = ContactMaterial()
    # system.GetContactContainer().RegisterAddContactCallback(cmaterial)
    return creporter, cmaterial

# for visualization
def visualize(system, creporter):
    application = chronoirr.ChIrrApp(system, 'PyChrono example: Collisions between objects',chronoirr.dimension2du(1024, 768))
    application.AddTypicalSky()
    # application.AddTypicalLogo(chrono.GetChronoDataFile('logo_pychrono_alpha.png'))
    application.AddTypicalCamera(chronoirr.vector3df(0, 14, -20))
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
    origin = [0, -10, 20]

    # initialize system
    system = chrono.ChSystemNSC()
    system.Set_G_acc(chrono.ChVectorD(0,-10,0))

    creporter, cmaterial = addGround(system, origin)
    system.GetContactContainer().RegisterAddContactCallback(cmaterial)

    # initialize tiles
    # for shape, 1st val represents good or bad. Remaining vals are x & z dimensions (in cm) and position
    # version = "god"
    # if version == "good":
    #     # plain rectangle, must have 0
    #     tile01 = Tile(system, [(1, 6, 18, origin[0], 1, origin[2])])
    # else:
    # f shape
    tile01 = Tile(system, [(0, 6, 18, origin[0], 1, origin[2])])
    # tile10 = Tile(system, [(0, 15, 5, origin[0], -8, origin[2]), (0, 4, 6.5, origin[0]-1, -8, origin[2] + 6), (0, 4.5, 7, origin[0]+5, -8, origin[2]+6)])
    pillar1 = Pillar(system, (10, -5, 30))


    visualize(system, creporter)