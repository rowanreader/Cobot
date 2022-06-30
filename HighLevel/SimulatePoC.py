### PROOF OF CONCEPT FOR PHYSICS ENGINE
# import pychrono as chrono

import pychrono.core as chrono
import pychrono.irrlicht as chronoirr
import numpy as np

chrono.SetChronoDataPath('/home/jacqueline/anaconda3/pkgs/pychrono-7.0.0-py38_0/share/chrono/data/')

class ContactReporter(chrono.ReportContactCallback):
    def __init__(self, box):
        self.m_box = box
        super().__init__()

    def OnReportContact(self,
                        pA,
                        pB,
                        plane_coord,
                        distance,
                        eff_radius,
                        cforce,
                        ctorque,
                        modA,
                        modB):
        bodyA = chrono.CastToChBody(modA)
        bodyB = chrono.CastToChBody(modB)
        # if (bodyA == self.m_box):
        #     print("       ", pA.x, pA.y, pA.z)
        # elif (bodyB == self.m_box):
        #     print("       ", pB.x, pB.y, pB.z)
        print("Collide!")
        return True

class ContactMaterial(chrono.AddContactCallback):
    def __init__(self):
        super().__init__()
    def OnAddContact(         self,
                              contactinfo,
                              material):
        # Downcast to appropriate composite material type
        mat = chrono.CastToChMaterialCompositeNSC(material)

        # Set different friction for left/right halfs
        if (contactinfo.vpA.z > 0) :
            friction =  0.3
        else:
            friction =  0.8
        mat.static_friction = friction
        mat.sliding_friction = friction

class Pillar:
    # x and y are pillar coordinates in local (floor) coords
    # floor X Y and Z are origin of floor in global coords
    # floor angle is angle floor is at
    def __init__(self, x, y, floorId, floorX, floorY, floorZ, floorAng):
        # hard coded stuff - divide by 100
        divide = 100
        self.height = 45/divide
        self.radius = 6/divide

        self.x = x/divide
        self.y = y/divide
        self.floorId = floorId # number id representing floor - acts as index for floor array
        self.globalX, self.globalY = self.transform(self.x,self.y, floorX/divide, floorY/divide, floorAng)
        self.globalZ = floorZ/divide

    # given an x and y in floor coords, and the global coords of floor origin, rotate about origin (of floor) by angle
    # z won't rotate
    # then translate to get global coords
    # return global coords
    def transform(self, x, y, floorX, floorY, angle):
        """Transforms x and y, which are in local coords on the floor, into global coords"""
        rotation = np.array([[np.cos(angle), np.sin(angle)],[-np.sin(angle), np.cos(angle)]])
        rotatedCoord = np.matmul([x,y], rotation)
        newCoord = rotatedCoord + [floorX, floorY]
        return newCoord[0], newCoord[1]


class Floor:
    # takes in ID, mass (g), coordinates of origin of floor in global, spot coordinates in local, and pieces def (?)
    def __init__(self, floorId, mass, origin, spots):
        divide = 10
        self.Id = floorId
        self.mass = mass
        self.spots = spots
        self.x = origin[0]/divide
        self.y = origin[1]/divide
        self.z = origin[2]/divide
        self.com = [origin[0], origin[1], origin[2]] # com is initially the origin


# make rigid body
my_system = chrono.ChSystemNSC()
wood = chrono.ChMaterialSurfaceNSC() # wood is a non-smooth
# set frictions (just googled wood)
wood.SetFriction(0.5)
wood.SetCompliance(0) # rigid body I think - no give
cardboard = chrono.ChMaterialSurfaceNSC()
cardboard.SetFriction(0.3)
cardboard.SetCompliance(0) # technically untrue

mboxtexture = chrono.ChTexture()
mboxtexture.SetTextureFilename(chrono.GetChronoDataFile('textures/concrete.jpg'))
mfloortexture = chrono.ChTexture()
mfloortexture.SetTextureFilename(chrono.GetChronoDataFile('textures/tire.png'))

# pillar placed on axe spot at lowest level. Axe is at (10, 10)
pillar1 = Pillar(-30, -7.5, 8, 10, 10, 2, np.pi/3)
floor1 = Floor(1, 20, [3, 3, 3], [23, 32])


bodyA = chrono.ChBodyEasyCylinder(pillar1.radius, pillar1.height/2, 10, True, True, wood)
# bodyA.SetMass(20) # grams CHECK LATER
# # set diagonals of inertia matrix
# bodyA.SetInertiaXX(chrono.ChVectorD(10,10,10))
# bodyA.SetName("Pillar1")
# # translate to global coords
# bodyA.SetPos(chrono.ChVectorD(pillar1.globalX, pillar1.globalY, pillar1.globalZ))
# # define area of existence for pillar
# # material, radius x, radius z, half length, position, rotation
# # radius is same in x and y directions
# # z position is mid point of pillar, so add half the height
# # no rotation - make identity
# bodyA.GetCollisionModel().AddCylinder(wood, pillar1.radius, pillar1.radius, pillar1.height/2,\
#                                       chrono.ChVectorD(pillar1.globalX, pillar1.globalY, pillar1.globalZ + pillar1.height/2),\
#                                       chrono.ChMatrix33D(1))
bodyA.SetBodyFixed(True) # doesn't move (even if collides?)
creporter = ContactReporter(bodyA)
cmaterial = ContactMaterial()
my_system.GetContactContainer().RegisterAddContactCallback(cmaterial)
myapplication = chronoirr.ChIrrApp(my_system, 'Tower Test', chronoirr.dimension2du(1024,768))
# bodyA.SetCollide(True) # can collide
# bodyA.SetPos(chrono.ChVectorD(0, 0, 0))
my_system.Add(bodyA)
# bodyA.GetAssets().push_back(mboxtexture)
# mboxasset = chrono.ChCylinderShape()
# p1 = 0
# mboxasset.GetCylinderGeometry().Size = chrono.ChVectorD(p1, p1 + pillar1.height, pillar1.radius)
# bodyA.AddAsset(mboxasset)


bodyB = chrono.ChBodyEasyBox(0.5, 0.1, 0.5, 10, True, True, cardboard)
# bodyB.SetMass(10)
# bodyB.SetInertiaXX(chrono.ChVectorD(10,10,10))
# bodyB.SetName("Floor 1")
bodyB.SetPos(chrono.ChVectorD(pillar1.globalX, pillar1.globalY + 5, pillar1.globalZ))
# bodyB.GetCollisionModel().AddBox(cardboard,.5,.01,.5,chrono.ChVectorD(pillar1.globalX, pillar1.globalY + pillar1.height/2, pillar1.globalZ) )
# bodyB.SetBodyFixed(False)
# bodyB.SetCollide(True)
# bodyB.SetPos(chrono.ChVectorD(pillar1.globalX, pillar1.globalY+ pillar1.height/2, pillar1.globalZ))
# bodyB.GetAssets().push_back(mfloortexture)
# mfloorasset = chrono.ChBoxShape()
# mfloorasset.GetBoxGeometry().Size = chrono.ChVectorD(.5,.01,.5)
# bodyB.AddAsset(mfloorasset)
my_system.Add(bodyB)




myapplication.AddTypicalSky()
myapplication.AddTypicalLogo()
myapplication.AddTypicalCamera(chronoirr.vector3df(0.6,0.6,0.8))
# myapplication.AddCamera(chronoirr.vector3df(0,0,0))
myapplication.AddTypicalLights()
myapplication.AssetBindAll();
myapplication.AssetUpdateAll();
myapplication.SetTimestep(0.02)
while(myapplication.GetDevice().run()):
    myapplication.BeginScene()
    myapplication.DrawAll()
    myapplication.DoStep()
    myapplication.EndScene()
    my_system.GetContactContainer().ReportAllContacts(creporter)
