To run: %run LoadCalibData.py
'''
import numpy as np
import matplotlib.pyplot as plt
import scipy.spatial.distance as distance
from mpl_toolkits.mplot3d import axes3d

data = np.loadtxt('data.txt')
reprojected3Dmap = np.zeros((data.shape[0], 4))

def calibrateCamera3D(data):
    A = np.zeros((4, 4))  # Pre-allocate matrix
    Ximg = np.zeros((4, 1))
    Yimg = np.zeros((4, 1))

    # finding a suitable "matrix A" whoose determinant is not zero
    isSuitableMatrix = False
    for i in range(0, data.shape[0]):
        A[0, :] = [data[i, 0], data[i, 1], data[i, 2], 1]
        Ximg[0, :] = [data[i, 3]]
        Yimg[0, :] = [data[i, 4]]
        for j in range(1, data.shape[0]):
            A[1, :] = [data[j, 0], data[j, 1], data[j, 2], 1]
            Ximg[1, :] = [data[j, 3]]
            Yimg[1, :] = [data[j, 4]]
            for k in range(2, data.shape[0]):
                A[2, :] = [data[k, 0], data[k, 1], data[k, 2], 1]
                Ximg[2, :] = [data[k, 3]]
                Yimg[2, :] = [data[k, 4]]
                for l in range(3, data.shape[0]):
                    A[3, :] = [data[l, 0], data[l, 1], data[l, 2], 1]
                    Ximg[3, :] = [data[l, 3]]
                    Yimg[3, :] = [data[l, 4]]
                    if np.linalg.det(A) != 0:
                        isSuitableMatrix = True
                        break
                if isSuitableMatrix:
                    break
            if isSuitableMatrix:
                break
        if isSuitableMatrix:
            break

    A_inverse = np.linalg.inv(A)
    ProjectionMatrix = np.zeros((4, 4))
    ProjectionMatrix[0, :] = np.squeeze(np.asarray(A_inverse.dot(Ximg)))
    ProjectionMatrix[1, :] = np.squeeze(np.asarray(A_inverse.dot(Yimg)))
    ProjectionMatrix[2, :] = np.squeeze(np.asarray(A_inverse.dot([1, 1, 1, 1])))
    ProjectionMatrix[3, :] = [0, 0, 0, 1]
    return ProjectionMatrix

def visualiseCameraCalibration3D(data, P):
    fig = plt.figure()
    ax = fig.gca()
    ax.plot(data[:, 3], data[:, 4], 'r.')

    renderedImage = np.zeros((data.shape[0], 2))
    for i in range(0, data.shape[0]):
        Prwl = [data[i,0],data[i,1],data[i,2],1]
        Cam = P.dot(Prwl)
        Xcam=Cam[0]/Cam[2]
        Ycam = Cam[1] / Cam[2]
        renderedImage[i,:]=[Xcam,Ycam]

        inverse_P=np.linalg.inv(P)
        reprojected3Dpoint=inverse_P.dot(Cam)
        reprojected3Dmap[i,:]= np.squeeze(np.asarray(reprojected3Dpoint))
    ax.plot(renderedImage[:, 0], renderedImage[:, 1], 'r.')
    fig = plt.figure()
    ax = fig.gca(projection="3d")
    ax.plot(reprojected3Dmap[:, 0], reprojected3Dmap[:, 1], reprojected3Dmap[:, 2], 'k.')

def evaluateCameraCalibration3D(data, P):
    print "......Projected 3d image......."
    points = np.array(reprojected3Dmap)
    dist = distance.pdist(points)
    print "max distance : ", dist.max()
    print "min distance : ",dist.min()
    print "mean distance : ",np.mean(dist)
    print "variance : ",np.var(dist)

    print "......Given 3d image(real)......."
    Real3Dvalues = np.zeros((data.shape[0], 3))
    for i in range(0, data.shape[0]):
        Real3Dvalues[0, :] =[data[i,0],data[i,1],data[i,2]]
    points = np.array(Real3Dvalues)
    dist = distance.pdist(points)
    print "max distance : ", dist.max()
    print "min distance : ", dist.min()
    print "mean distance : ", np.mean(dist)
    print "variance : ", np.var(dist)

P=calibrateCamera3D(data)
visualiseCameraCalibration3D(data, P)
evaluateCameraCalibration3D(data, P)

#plotting real values
fig = plt.figure()
ax = fig.gca(projection="3d")
ax.plot(data[:,0], data[:,1], data[:,2],'k.')

fig = plt.figure()
ax = fig.gca()
ax.plot(data[:,3], data[:,4],'r.')

plt.show()