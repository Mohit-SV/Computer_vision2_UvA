import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import random
from scipy.spatial import procrustes
import open3d as o3d
import scipy
import open3d as o3d
import matplotlib
import matplotlib.pyplot as plt
from scipy.linalg import orthogonal_procrustes
from create_PVM import create_PVM

#################################################################################################################
######################### STRUCTURE FROM MOTION WITH PROVIDED POINT-VIEW-MATRIX #################################
#################################################################################################################

# return 3d pointcloud for dense point-view-matrix, no stitching is needed
def get_pointcloud(matrix):
    mean = np.mean(matrix, axis=1)
    mean = mean[:, np.newaxis]
    D = matrix- mean
    
    U, W, V = np.linalg.svd(D)
    V = V.T
    W = np.sqrt(np.array(np.diag(W[:3]), dtype=np.float))
    U = U[:, :3]
    V = V[:, :3]

    # U W V.T = U sqrt(W) sqrt(W) V.T
    # [motion] [structure] = [U sqrt(W)][sqrt(W) V.T]
    M = np.dot(U, W)
    S = np.dot(W, V.T)

    return S, M

matrix = np.loadtxt('PointViewMatrix.txt')
S, M = get_pointcloud(matrix)
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(S.T)
o3d.visualization.draw_geometries([pcd])  

#################################################################################################################
################################## STRUCTURE FROM MOTION WITH SPARSE CUSTOM POINT-VIEW MATRIX ###################
#################################################################################################################

# alter function so that we get the transformation parameters as well, 
# function from scripy.linalg procrustes, but then to return T ans s as well. 
def procrustes(data1, data2):

    mtx1 = np.array(data1, dtype=np.double, copy=True)
    mtx2 = np.array(data2, dtype=np.double, copy=True)

    if mtx1.ndim != 2 or mtx2.ndim != 2:
        raise ValueError("Input matrices must be two-dimensional")
    if mtx1.shape != mtx2.shape:
        raise ValueError("Input matrices must be of same shape")
    if mtx1.size == 0:
        raise ValueError("Input matrices must be >0 rows and >0 cols")

    # translate all the data to the origin
    mtx1 -= np.mean(mtx1, 0)
    mtx2 -= np.mean(mtx2, 0)

    norm1 = np.linalg.norm(mtx1)
    norm2 = np.linalg.norm(mtx2)

    if norm1 == 0 or norm2 == 0:
        raise ValueError("Input matrices must contain >1 unique points")

    # change scaling of data (in rows) such that trace(mtx*mtx') = 1
    mtx1 /= norm1
    mtx2 /= norm2

    # transform mtx2 to minimize disparity
    R, s = orthogonal_procrustes(mtx1, mtx2)
    mtx2 = np.dot(mtx2, R.T) * s

    # measure the dissimilarity between the two datasets
    disparity = np.sum(np.square(mtx1 - mtx2))

    return mtx2, R, s

# obtain dense blocks from sparse point-view-matrix for a certain number
# of frames, in this assignment we experiment with 3 and 4 frames
# A dense block consists of points which only occur in all the n frames
def get_dense(frames):
    index = []
    for p in range(len(frames[0])):
        flag = 1
        for i in range(0,len(frames)-1,2):     
            # no interest point at location 0,0 so its safe for this dataset
            if int(frames[i, p]) == 0 and int(frames[i+1,p]) == 0:              
                flag = 0

        if flag == 1:
            index.append(True)
        else:
            index.append(False)

    return frames[:, index], index


def remove_ambiguity(S,M):
    # slides: A L A.T = I 
    # so L = inv(A) I inv(A.T) 
    # inv(a_i) * inv(a_i.T)
    A = np.linalg.pinv(M)
    AT =np.concatenate([[np.linalg.pinv(M[i:i+2]).T] for i in range(0, len(M), 2)], axis=1).squeeze()
    L = np.dot(A, AT)

    # perform choslesky as in slides and compute new S and M
    C = np.linalg.cholesky(L)
    M = np.dot(M,C)
    S = np.dot(np.linalg.inv(C),S)

    #################### CHECKS ###################################
    #for i in range(0, len(M)-2, 2):
    #    print(np.dot(M[i].T, M[i]))        # this is ca. 1
    #    print(np.dot(M[i+1].T, M[i+1]))    # this is ca. 1
    #    print(np.dot(M[i].T,M[i+1]))       # this is ca. 0
    #    print("----------")
    #################################################################

    return S, M

# Return structure and motion matrices for every dense block
def motion_structure(dense):

    # remove affine ambiguity by 
    remove_amb = False

    # follow steps from slides

    # normalize
    mean = np.mean(dense, axis=1)
    mean = mean[:, np.newaxis]
    D = dense - mean
    
    U, W, V = np.linalg.svd(D)

    V = V.T
    W = np.sqrt(np.array(np.diag(W[:3]), dtype=np.float))
    U = U[:, :3]
    V = V[:, :3]

    # U W V.T = U sqrt(W) | sqrt(W) V.T with  part before | 
    # being motion and part after the | being structure

    # obtain motion
    M = np.dot(U, W)

    # obtain structure
    S = np.dot(W, V.T)

    # remove ambiguity by enforcing that the rows of M 
    # are perpendicular, i.e. np.dot(row i.T, row i) = 1
    # np.dot(row j.T row j) = 1 and np.dot(row i.T, row j) = 0
    if remove_amb:
        S, M = remove_ambiguity(S,M)
              
    return S, M

# Create dense blocks from sparse point-view-matrix, for every
# block compute S and M and stitch all views together.
def get_pointcloud(frames, matrix):

    # iteratively go over all images/views
    for f in range(0,len(matrix)-2*frames, 2*frames):

        # get dense matrix and perform factarization
        # to obtain structure and motion matrices
        # S are the 3d point coordinates
        dense, index  = get_dense(matrix[f:f+6])
        S, M = motion_structure(dense)   

        # main view is the first image
        if f == 0:
            pointcloud = S
            S_prev = S
            prev_index = index         
            continue

        # get indices for matches, i.e get true indices
        # and only take indices of the other
        # if these are also true
        match_prev = np.array(index)[np.array(prev_index)]
        match = np.array(prev_index)[np.array(index)]

        # mtx2 contain the transformed points we want, only we need to transform
        # the entire S matrix.
        try:
            mtx2, R, s = procrustes(S[:, match].T, S_prev[:, match_prev].T)
            S = (np.dot(S.T, R) * s).T
        
            S_prev = S
            prev_index = index
            pointcloud = np.concatenate((pointcloud, S), axis=1)
        except:
            print("something is wrong")
            continue

    # scale z-axis a little bit for visualization purposes to better see
    # that the cloud is not flat
    pointcloud[2,:] = pointcloud[2,:]*2
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pointcloud.T)
    o3d.visualization.draw_geometries([pcd])

matrix, fundamental = create_PVM()
A = get_pointcloud(4, matrix)