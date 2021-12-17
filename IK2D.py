import numpy as np



def FK2D(lengths, angles):
    pass

def Jacobian(q, lengths):
    pass

def IK2D():
    pd = np.random.rand(2)*1000 # random destination

    qOld = 0
    q = 2*np.pi*np.random.rand(5)
    angles = [np.pi/5, np.pi/4, np.pi/3, np.pi/2, np.pi] # just random angles
    lengths = [1,1,1,1,1] # set all lengths to 1
    count = 0
    alpha = 0.2

    p, joints, endEffector = FK2D(q)
    while qOld - q > 0.01 and count < 100:
        count += 1
        diff = pd - p
        jacob = Jacobian(q, lengths)
        dq = np.matmul(np.linalg.pinv(jacob), diff)
        qOld = q

        q = q + alpha*dq
        p, joints, endEffector = FK2D(q)

        # plot