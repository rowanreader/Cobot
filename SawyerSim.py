# given initial state (forward kinematics), and goal destination, moves Sawyer robot
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as ani
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D

# given joints and end effector position, plot (also plot goal point if it's given)
def plot(joints, endEffector, goal=None):
    plt.clf()
    ax = plt.axes(projection='3d')
    ax.plot3D(joints[:,0], joints[:,1], joints[:,2], '-r')
    ax.plot3D(endEffector[:,0], endEffector[:,1], endEffector[:,2], '.-b')
    ax.plot3D(goal[0], goal[1], goal[2], '*g')
    plt.show()

def DH(a, alpha, d, th):
    T = [[np.cos(th),  -np.sin(th)*np.cos(alpha), np.sin(th)*np.sin(alpha),  a*np.cos(th)],
         [np.sin(th),  np.cos(th)*np.cos(alpha),  -np.cos(th)*np.sin(alpha), a*np.sin(th)],
         [0,                np.sin(alpha),                np.cos(alpha),                d],
         [0,                    0,                              0,                      1]]

    return T

# given angles of the Sawyer robot, finds end effector position
def FK(angles):
    j0 = np.array([0, 0, 0]) # anchor at origin
    d = [317, 192.5, 400, 168.5, 400, 136.3, 133.75] # link length
    a = [81, 0, 0, 0, 0, 0, 0]
    alpha = [-np.pi/2, -np.pi/2, -np.pi/2, -np.pi/2, -np.pi/2, -np.pi/2, 0]
    th = [0, 3*np.pi/2, 0, np.pi, 0, np.pi, 3*np.pi/2] # added to angles
    # get all T matrices
    T1 = DH(a[0], alpha[0], d[0], th[0] + angles[0])
    T2 = DH(a[1], alpha[1], d[1], th[1] + angles[1])
    T3 = DH(a[2], alpha[2], d[2], th[2] + angles[2])
    T4 = DH(a[3], alpha[3], d[3], th[3] + angles[3])
    T5 = DH(a[4], alpha[4], d[4], th[4] + angles[4])
    T6 = DH(a[5], alpha[5], d[5], th[5] + angles[5])
    TP = DH(a[6], alpha[6], d[6], th[6] + angles[6])

    id = [0, 0, 0, 1] # extracts last col
    j1 = np.matmul(T1, id) # will be a 1D array
    j1 = j1[0:3] # just 1st 3 nums

    j1_ = np.matmul(T1, [-81, 0, 0, 1])
    j1_ = j1_[0:3]

    T12 = np.matmul(T1, T2)
    j2 = np.matmul(T12, id)
    j2 = j2[0:3]

    T13 = np.matmul(T12, T3)
    j3 = np.matmul(T13, id)
    j3 = j3[0:3]

    T14 = np.matmul(T13, T4)
    j4 = np.matmul(T14, id)
    j4 = j4[0:3]

    T15 = np.matmul(T14, T5)
    j5 = np.matmul(T15, id)
    j5 = j5[0:3]

    T16 = np.matmul(T15, T6)
    j6 = np.matmul(T16, id)
    j6 = j6[0:3]

    # technically this is the end effecctor and a joint
    T1P = np.matmul(T16, TP)
    jP = np.matmul(T1P, id)
    jP = jP[0:3]

    joints =np.array([j0, j1_, j1, j2, j3, j4, j5, j6, jP])

    claw = 40 # size of claw lengths (all identical)
    A = [claw, 0, 0, 1]
    B = [claw, 0, claw, 1]
    C = [-claw, 0, 0, 1]
    D = [-claw, 0, claw, 1]

    A = np.matmul(T1P, A)
    B = np.matmul(T1P, B)
    C = np.matmul(T1P, C)
    D = np.matmul(T1P, D)

    actuators = np.array([B[0:3], A[0:3], C[0:3], D[0:3]])

    return jP, joints, actuators

# given angles, calculates Jacobian
def Jacobian(angles):
    th1 = angles[0]
    th2 = angles[1]
    th3 = angles[2]
    th4 = angles[3]
    th5 = angles[4]
    th6 = angles[5]
    l1 = 317
    l2 = 192.5
    l3 = 400
    l4 = 168.5
    l5 = 400
    l6 = 136.3
    l7 = 133.75
    jacob = [[l4*(np.cos(th1)*np.cos(th3) + np.sin(th1)*np.sin(th2)*np.sin(th3)) - l5*(np.cos(th2)*np.cos(th4)*np.sin(th1) - np.cos(th1)*np.sin(th3)*np.sin(th4) + np.cos(th3)*np.sin(th1)*np.sin(th2)*np.sin(th4)) - 81*np.sin(th1) + l6*(np.sin(th5)*(np.cos(th2)*np.sin(th1)*np.sin(th4) + np.cos(th1)*np.cos(th4)*np.sin(th3) - np.cos(th3)*np.cos(th4)*np.sin(th1)*np.sin(th2)) - np.cos(th5)*(np.cos(th1)*np.cos(th3) + np.sin(th1)*np.sin(th2)*np.sin(th3))) - l7*(np.cos(th6)*(np.cos(th2)*np.cos(th4)*np.sin(th1) - np.cos(th1)*np.sin(th3)*np.sin(th4) + np.cos(th3)*np.sin(th1)*np.sin(th2)*np.sin(th4)) + np.sin(th6)*(np.cos(th5)*(np.cos(th2)*np.sin(th1)*np.sin(th4) + np.cos(th1)*np.cos(th4)*np.sin(th3) - np.cos(th3)*np.cos(th4)*np.sin(th1)*np.sin(th2)) + np.sin(th5)*(np.cos(th1)*np.cos(th3) + np.sin(th1)*np.sin(th2)*np.sin(th3)))) - l3*np.cos(th2)*np.sin(th1), l6*(np.sin(th5)*(np.cos(th1)*np.sin(th2)*np.sin(th4) + np.cos(th1)*np.cos(th2)*np.cos(th3)*np.cos(th4)) + np.cos(th1)*np.cos(th2)*np.cos(th5)*np.sin(th3)) - l7*(np.cos(th6)*(np.cos(th1)*np.cos(th4)*np.sin(th2) - np.cos(th1)*np.cos(th2)*np.cos(th3)*np.sin(th4)) + np.sin(th6)*(np.cos(th5)*(np.cos(th1)*np.sin(th2)*np.sin(th4) + np.cos(th1)*np.cos(th2)*np.cos(th3)*np.cos(th4)) - np.cos(th1)*np.cos(th2)*np.sin(th3)*np.sin(th5))) - l5*(np.cos(th1)*np.cos(th4)*np.sin(th2) - np.cos(th1)*np.cos(th2)*np.cos(th3)*np.sin(th4)) - l3*np.cos(th1)*np.sin(th2) - l4*np.cos(th1)*np.cos(th2)*np.sin(th3), l6*(np.cos(th5)*(np.sin(th1)*np.sin(th3) + np.cos(th1)*np.cos(th3)*np.sin(th2)) + np.cos(th4)*np.sin(th5)*(np.cos(th3)*np.sin(th1) - np.cos(th1)*np.sin(th2)*np.sin(th3))) - l4*(np.sin(th1)*np.sin(th3) + np.cos(th1)*np.cos(th3)*np.sin(th2)) + l7*(np.sin(th6)*(np.sin(th5)*(np.sin(th1)*np.sin(th3) + np.cos(th1)*np.cos(th3)*np.sin(th2)) - np.cos(th4)*np.cos(th5)*(np.cos(th3)*np.sin(th1) - np.cos(th1)*np.sin(th2)*np.sin(th3))) + np.cos(th6)*np.sin(th4)*(np.cos(th3)*np.sin(th1) - np.cos(th1)*np.sin(th2)*np.sin(th3))) + l5*np.sin(th4)*(np.cos(th3)*np.sin(th1) - np.cos(th1)*np.sin(th2)*np.sin(th3)),   l7*(np.cos(th6)*(np.cos(th4)*np.sin(th1)*np.sin(th3) - np.cos(th1)*np.cos(th2)*np.sin(th4) + np.cos(th1)*np.cos(th3)*np.cos(th4)*np.sin(th2)) + np.cos(th5)*np.sin(th6)*(np.sin(th1)*np.sin(th3)*np.sin(th4) + np.cos(th1)*np.cos(th2)*np.cos(th4) + np.cos(th1)*np.cos(th3)*np.sin(th2)*np.sin(th4))) + l5*(np.cos(th4)*np.sin(th1)*np.sin(th3) - np.cos(th1)*np.cos(th2)*np.sin(th4) + np.cos(th1)*np.cos(th3)*np.cos(th4)*np.sin(th2)) - l6*np.sin(th5)*(np.sin(th1)*np.sin(th3)*np.sin(th4) + np.cos(th1)*np.cos(th2)*np.cos(th4) + np.cos(th1)*np.cos(th3)*np.sin(th2)*np.sin(th4)),   l6*(np.cos(th5)*(np.cos(th4)*np.sin(th1)*np.sin(th3) - np.cos(th1)*np.cos(th2)*np.sin(th4) + np.cos(th1)*np.cos(th3)*np.cos(th4)*np.sin(th2)) + np.sin(th5)*(np.cos(th3)*np.sin(th1) - np.cos(th1)*np.sin(th2)*np.sin(th3))) + l7*np.sin(th6)*(np.sin(th5)*(np.cos(th4)*np.sin(th1)*np.sin(th3) - np.cos(th1)*np.cos(th2)*np.sin(th4) + np.cos(th1)*np.cos(th3)*np.cos(th4)*np.sin(th2)) - np.cos(th5)*(np.cos(th3)*np.sin(th1) - np.cos(th1)*np.sin(th2)*np.sin(th3))), -l7*(np.sin(th6)*(np.sin(th1)*np.sin(th3)*np.sin(th4) + np.cos(th1)*np.cos(th2)*np.cos(th4) + np.cos(th1)*np.cos(th3)*np.sin(th2)*np.sin(th4)) + np.cos(th6)*(np.cos(th5)*(np.cos(th4)*np.sin(th1)*np.sin(th3) - np.cos(th1)*np.cos(th2)*np.sin(th4) + np.cos(th1)*np.cos(th3)*np.cos(th4)*np.sin(th2)) + np.sin(th5)*(np.cos(th3)*np.sin(th1) - np.cos(th1)*np.sin(th2)*np.sin(th3)))), 0],
    [81*np.cos(th1) + l5*(np.sin(th1)*np.sin(th3)*np.sin(th4) + np.cos(th1)*np.cos(th2)*np.cos(th4) + np.cos(th1)*np.cos(th3)*np.sin(th2)*np.sin(th4)) + l4*(np.cos(th3)*np.sin(th1) - np.cos(th1)*np.sin(th2)*np.sin(th3)) + l6*(np.sin(th5)*(np.cos(th4)*np.sin(th1)*np.sin(th3) - np.cos(th1)*np.cos(th2)*np.sin(th4) + np.cos(th1)*np.cos(th3)*np.cos(th4)*np.sin(th2)) - np.cos(th5)*(np.cos(th3)*np.sin(th1) - np.cos(th1)*np.sin(th2)*np.sin(th3))) + l7*(np.cos(th6)*(np.sin(th1)*np.sin(th3)*np.sin(th4) + np.cos(th1)*np.cos(th2)*np.cos(th4) + np.cos(th1)*np.cos(th3)*np.sin(th2)*np.sin(th4)) - np.sin(th6)*(np.cos(th5)*(np.cos(th4)*np.sin(th1)*np.sin(th3) - np.cos(th1)*np.cos(th2)*np.sin(th4) + np.cos(th1)*np.cos(th3)*np.cos(th4)*np.sin(th2)) + np.sin(th5)*(np.cos(th3)*np.sin(th1) - np.cos(th1)*np.sin(th2)*np.sin(th3)))) + l3*np.cos(th1)*np.cos(th2),                                             -np.sin(th1)*(l3*np.sin(th2) + l4*np.cos(th2)*np.sin(th3) + l5*np.cos(th4)*np.sin(th2) - l5*np.cos(th2)*np.cos(th3)*np.sin(th4) - l6*np.cos(th2)*np.cos(th5)*np.sin(th3) + l7*np.cos(th4)*np.cos(th6)*np.sin(th2) - l6*np.sin(th2)*np.sin(th4)*np.sin(th5) - l6*np.cos(th2)*np.cos(th3)*np.cos(th4)*np.sin(th5) - l7*np.cos(th2)*np.cos(th3)*np.cos(th6)*np.sin(th4) - l7*np.cos(th2)*np.sin(th3)*np.sin(th5)*np.sin(th6) + l7*np.cos(th5)*np.sin(th2)*np.sin(th4)*np.sin(th6) + l7*np.cos(th2)*np.cos(th3)*np.cos(th4)*np.cos(th5)*np.sin(th6)), l4*(np.cos(th1)*np.sin(th3) - np.cos(th3)*np.sin(th1)*np.sin(th2)) - l6*(np.cos(th5)*(np.cos(th1)*np.sin(th3) - np.cos(th3)*np.sin(th1)*np.sin(th2)) + np.cos(th4)*np.sin(th5)*(np.cos(th1)*np.cos(th3) + np.sin(th1)*np.sin(th2)*np.sin(th3))) - l7*(np.sin(th6)*(np.sin(th5)*(np.cos(th1)*np.sin(th3) - np.cos(th3)*np.sin(th1)*np.sin(th2)) - np.cos(th4)*np.cos(th5)*(np.cos(th1)*np.cos(th3) + np.sin(th1)*np.sin(th2)*np.sin(th3))) + np.cos(th6)*np.sin(th4)*(np.cos(th1)*np.cos(th3) + np.sin(th1)*np.sin(th2)*np.sin(th3))) - l5*np.sin(th4)*(np.cos(th1)*np.cos(th3) + np.sin(th1)*np.sin(th2)*np.sin(th3)), - l5*(np.cos(th2)*np.sin(th1)*np.sin(th4) + np.cos(th1)*np.cos(th4)*np.sin(th3) - np.cos(th3)*np.cos(th4)*np.sin(th1)*np.sin(th2)) - l7*(np.cos(th6)*(np.cos(th2)*np.sin(th1)*np.sin(th4) + np.cos(th1)*np.cos(th4)*np.sin(th3) - np.cos(th3)*np.cos(th4)*np.sin(th1)*np.sin(th2)) - np.cos(th5)*np.sin(th6)*(np.cos(th2)*np.cos(th4)*np.sin(th1) - np.cos(th1)*np.sin(th3)*np.sin(th4) + np.cos(th3)*np.sin(th1)*np.sin(th2)*np.sin(th4))) - l6*np.sin(th5)*(np.cos(th2)*np.cos(th4)*np.sin(th1) - np.cos(th1)*np.sin(th3)*np.sin(th4) + np.cos(th3)*np.sin(th1)*np.sin(th2)*np.sin(th4)), - l6*(np.cos(th5)*(np.cos(th2)*np.sin(th1)*np.sin(th4) + np.cos(th1)*np.cos(th4)*np.sin(th3) - np.cos(th3)*np.cos(th4)*np.sin(th1)*np.sin(th2)) + np.sin(th5)*(np.cos(th1)*np.cos(th3) + np.sin(th1)*np.sin(th2)*np.sin(th3))) - l7*np.sin(th6)*(np.sin(th5)*(np.cos(th2)*np.sin(th1)*np.sin(th4) + np.cos(th1)*np.cos(th4)*np.sin(th3) - np.cos(th3)*np.cos(th4)*np.sin(th1)*np.sin(th2)) - np.cos(th5)*(np.cos(th1)*np.cos(th3) + np.sin(th1)*np.sin(th2)*np.sin(th3))), -l7*(np.sin(th6)*(np.cos(th2)*np.cos(th4)*np.sin(th1) - np.cos(th1)*np.sin(th3)*np.sin(th4) + np.cos(th3)*np.sin(th1)*np.sin(th2)*np.sin(th4)) - np.cos(th6)*(np.cos(th5)*(np.cos(th2)*np.sin(th1)*np.sin(th4) + np.cos(th1)*np.cos(th4)*np.sin(th3) - np.cos(th3)*np.cos(th4)*np.sin(th1)*np.sin(th2)) + np.sin(th5)*(np.cos(th1)*np.cos(th3) + np.sin(th1)*np.sin(th2)*np.sin(th3)))), 0],
     [0, l4*np.sin(th2)*np.sin(th3) - l5*np.cos(th2)*np.cos(th4) - l3*np.cos(th2) - l7*np.cos(th2)*np.cos(th4)*np.cos(th6) - l5*np.cos(th3)*np.sin(th2)*np.sin(th4) - l6*np.cos(th5)*np.sin(th2)*np.sin(th3) + l6*np.cos(th2)*np.sin(th4)*np.sin(th5) - l6*np.cos(th3)*np.cos(th4)*np.sin(th2)*np.sin(th5) - l7*np.cos(th3)*np.cos(th6)*np.sin(th2)*np.sin(th4) - l7*np.cos(th2)*np.cos(th5)*np.sin(th4)*np.sin(th6) - l7*np.sin(th2)*np.sin(th3)*np.sin(th5)*np.sin(th6) + l7*np.cos(th3)*np.cos(th4)*np.cos(th5)*np.sin(th2)*np.sin(th6), -np.cos(th2)*(l4*np.cos(th3) - l6*np.cos(th3)*np.cos(th5) + l5*np.sin(th3)*np.sin(th4) + l6*np.cos(th4)*np.sin(th3)*np.sin(th5) + l7*np.cos(th6)*np.sin(th3)*np.sin(th4) - l7*np.cos(th3)*np.sin(th5)*np.sin(th6) - l7*np.cos(th4)*np.cos(th5)*np.sin(th3)*np.sin(th6)),                                                                                                                                                      l5*np.sin(th2)*np.sin(th4) + l5*np.cos(th2)*np.cos(th3)*np.cos(th4) + l6*np.cos(th4)*np.sin(th2)*np.sin(th5) + l7*np.cos(th6)*np.sin(th2)*np.sin(th4) + l7*np.cos(th2)*np.cos(th3)*np.cos(th4)*np.cos(th6) - l6*np.cos(th2)*np.cos(th3)*np.sin(th4)*np.sin(th5) - l7*np.cos(th4)*np.cos(th5)*np.sin(th2)*np.sin(th6) + l7*np.cos(th2)*np.cos(th3)*np.cos(th5)*np.sin(th4)*np.sin(th6), l6*np.cos(th5)*np.sin(th2)*np.sin(th4) - l6*np.cos(th2)*np.sin(th3)*np.sin(th5) + l6*np.cos(th2)*np.cos(th3)*np.cos(th4)*np.cos(th5) + l7*np.cos(th2)*np.cos(th5)*np.sin(th3)*np.sin(th6) + l7*np.sin(th2)*np.sin(th4)*np.sin(th5)*np.sin(th6) + l7*np.cos(th2)*np.cos(th3)*np.cos(th4)*np.sin(th5)*np.sin(th6), l7*np.cos(th4)*np.sin(th2)*np.sin(th6) - l7*np.cos(th2)*np.cos(th3)*np.sin(th4)*np.sin(th6) + l7*np.cos(th2)*np.cos(th6)*np.sin(th3)*np.sin(th5) - l7*np.cos(th5)*np.cos(th6)*np.sin(th2)*np.sin(th4) - l7*np.cos(th2)*np.cos(th3)*np.cos(th4)*np.cos(th5)*np.cos(th6), 0]]
    return jacob

def IK(goal, q):

    moviewriter = ani.PillowWriter()
    fig = plt.figure(1)
    p, joints, endEffector = FK(q)

    qOld = [0,0,0,0,0,0,0] # default configuration
    count = 0
    alpha = 0.2
    with moviewriter.saving(fig, 'IKsimulation.gif', dpi=100):
        while np.linalg.norm(np.subtract(qOld, q)) > 0.01 and count < 70:
            # print(np.linalg.norm(np.subtract(qOld, q)))
            count = count + 1

            diff = np.subtract(goal, p)
            jacob = Jacobian(q)
            dq = np.matmul(np.linalg.pinv(jacob), diff)
            qOld = q
            q = q + alpha * dq
            p, joints, endEffector = FK(q)
            # plot(joints, endEffector, goal)
            plt.clf()
            ax = plt.axes(projection='3d')
            ax.plot3D(joints[:, 0], joints[:, 1], joints[:, 2], '.-r')
            ax.plot3D(endEffector[:, 0], endEffector[:, 1], endEffector[:, 2], '.-b')
            ax.plot3D(goal[0], goal[1], goal[2], '*g')

            moviewriter.grab_frame()


    moviewriter.finish()
    print(count)




if __name__ == "__main__":
    goal = [100,100,1000]
    q = [1,1,1,1,1,1,1] # start configuration
    IK(goal, q)


