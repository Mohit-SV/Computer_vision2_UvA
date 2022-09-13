import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import random
from scipy.spatial import procrustes
import open3d as o3d


##################################################################################################################
######################################### FUNDAMENTAL MATRIX  ####################################################
##################################################################################################################

# select desired method for estimating F
#flag = 'regular'
#flag = 'norm'
flag = 'ransac'

# find SIFT features and matches between the two images
def get_SIFT(img1, img2):
	
    # from https://docs.opencv.org/4.x/d1/de0/tutorial_py_feature_homography.html
    # and https://docs.opencv.org/3.4/dc/dc3/tutorial_py_matcher.html

	# Initiate SIFT detector
	sift = cv.SIFT_create()
	# find the keypoints and descriptors with SIFT
	kp1, des1 = sift.detectAndCompute(img1,None)
	kp2, des2 = sift.detectAndCompute(img2,None)

	# FLANN parameters
	FLANN_INDEX_KDTREE = 1
	index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
	search_params = dict(checks=50)   # or pass empty dictionary
	flann = cv.FlannBasedMatcher(index_params,search_params)
	matches = flann.knnMatch(des1,des2,k=2)

	good = []

	# good points are points below distance threshold
	for i,(m,n) in enumerate(matches):
	    if m.distance < 0.1*n.distance:
	    	good.append(m)


	# int32 for using the function drawepipolar line
	p1 = np.int32([ kp1[m.queryIdx].pt for m in good ])
	p2 = np.int32([ kp2[m.trainIdx].pt for m in good ])

	return p1, p2

# Normalize images to have ca. 0 mean and sqrt(2) std 
def normalization(points):
	# get mx and my 
	m = np.mean(points, axis=0)

	# last dimension is now always zero 
	tmp1 = (points-m)**2	
	
	d = np.mean(np.sqrt(np.sum(tmp1, axis=1)))

	T = np.array([[(2 ** 0.5)/d, 0, -m[0] * (2 ** 0.5)/d],[0, (2 ** 0.5)/d, -m[1] * (2 ** 0.5)/d],[0, 0, 1]])

	# if you do np.dot(points, T) you get different results??
	new_points = np.dot(T, points.T).T

	########################### CHECKS ###########################

	# check mean is actually 0, here almost zero i.e. e-18
	#print(np.mean(new_points, axis=0))

	# check avg distance to mean is sqrt(2)
	#print(np.std(new_points, axis=0))

	# check inverse of transformation is the same as original
	#print(np.dot(np.linalg.inv(T), new_points.T).T[0])

	##############################################################

	return new_points, T
   

def eightpoint(p1, p2):

    x1 = p1[:, 0]
    y1 = p1[:, 1]
    x2 = p2[:, 0]
    y2 = p2[:, 1]

    # construct nx9 matrix A
    A = np.array([x1 * x2, x1 * y2, x1, y1 * x2, y1 * y2, y1, x2, y2, np.ones(len(y1))]).T

    # Find SVD of A
    U, D, V = np.linalg.svd(A)

    # get smallest singular value, i.e in last position
    F = V[-1][:, np.newaxis].reshape(3,3)
    
    U_new, D_new, V_new = np.linalg.svd(F.T)

    # set smallest singular value in F to zero   
    D_new[-1] = 0
 
    # recompute F with rank 2
    F_new = (np.dot(U_new, np.dot(np.diag(D_new), V_new)))

    return F_new

def eightpoint_norm(p1, p2):

    # remove this line for no normalization 
    p1, T1 = normalization(p1)
    p2, T2 = normalization(p2)

    x1 = p1[:, 0]
    y1 = p1[:, 1]
    x2 = p2[:, 0]
    y2 = p2[:, 1]
    ones = np.ones(len(y1))

    # construct nx9 matrix A
    A = np.array([x1 * x2, x1 * y2, x1, y1 * x2, y1 * y2, y1, x2, y2, ones]).T

    # Find SVD of A
    U, D, V = np.linalg.svd(A)

    # get smallest singular value, i.e in last position
    F = V[-1][:, np.newaxis].reshape(3,3)
    
    U_new, D_new, V_new = np.linalg.svd(F.T)

    # set smallest singular value in F to zero   
    D_new[-1] = 0
 
    # recompute F with rank 2
    F_new = (np.dot(U_new, np.dot(np.diag(D_new), V_new)))

    # denormalization
    F_new = np.dot(T2.T, np.dot(F_new, T1))

    return F_new


def RANSAC(p11, p22):

    best = 0

    best_indices = 0
    for i in range(500):
		    tmp_best = []
        
		    randomlist = random.sample(range(0, p11.shape[0]), 8) 
		    #randomlist = np.random.choice(p11.shape[0], 8, replace=False)
		    F = eightpoint_norm(p11[randomlist,:],p22[randomlist,:])
				
			# "The suggested threshold would be somewhere between 1 to 10 (1~3 pixel error), 
			# depending on the size/resolution/quality of your image pairs"-> source:stackoverflow
            # However for consecutive views the distance can be set much lower
		    total_count = 0
		    for j in range(len(p11)):
		     
		        tmp1  = (np.dot(p22[j], np.dot(F, p11[j].T)))**2       	
		        tmp2 = np.dot(p11[j], F)[0]**2 + np.dot(p11[j],F)[1]**2
		        tmp3 = np.dot(p22[j], F)[0]**2 + np.dot(p22[j],F)[1]**2

		        sampson = tmp1/(tmp2+tmp3)
		      
		        if sampson < 0.00005:
		        				total_count = total_count + 1
		        				tmp_best.append(j)

		    if total_count > best:
		        			best = total_count
		        			best_indices = tmp_best


    F = eightpoint_norm(p11[best_indices,:],p22[best_indices,:])   
    return F, p11[best_indices,:], p22[best_indices,:]			

# read in images 
img1 = cv.imread("Data/House/House/frame00000001.png",0)
img2 = cv.imread("Data/House/House/frame00000002.png",0)

p1, p2 = get_SIFT(img1, img2)


# desired method
if flag == 'regular':
		F = eightpoint(np.insert(p1, 2, 1, axis=1), np.insert(p2, 2, 1, axis=1))
		pts2 = p2
		pts1 = p1
		p1 = np.insert(p1, 2, 1, axis=1)
		p2 = np.insert(p2, 2, 1, axis=1)

if flag == 'norm':
		F = eightpoint_norm(np.insert(p1, 2, 1, axis=1), np.insert(p2, 2, 1, axis=1))
		pts2 = p2
		pts1 = p1
		p1 = np.insert(p1, 2, 1, axis=1)
		p2 = np.insert(p2, 2, 1, axis=1)

if flag == 'ransac':
		F, p1, p2 = RANSAC(np.insert(p1, 2, 1, axis=1), np.insert(p2, 2, 1, axis=1))
		pts2 = p2[:, :2]
		pts1 = p1[:, :2]

# calculate epipolar lines from scratch
def epipolar_lines(F, p1, p2):
	# we can compute the epipolar lines l`= F.T p and l= F p'
    l2 = p1 @ F.T 
    l1 = p2 @ F

	# check epipolar constaints
    total = []
    for i in range(len(p1)):
    	total.append(p1[i].transpose() @ F @ p2[i])   
    print(np.mean(total))
    return l1, l2

l1, l2 = epipolar_lines(F, p1, p2)

# use drawing functions from https://docs.opencv.org/4.x/da/de9/tutorial_py_epipolar_geometry.html
#for easy pretty visualization
def drawlines(img1,img2,lines,pts1,pts2):
    ''' img1 - image on which we draw the epilines for the points in img2
        lines - corresponding epilines '''
    r,c = img1.shape
    img1 = cv.cvtColor(img1,cv.COLOR_GRAY2BGR)
    img2 = cv.cvtColor(img2,cv.COLOR_GRAY2BGR)
    for r,pt1,pt2 in zip(lines,pts1,pts2):
        color = tuple(np.random.randint(0,255,3).tolist())
        x0,y0 = map(int, [0, -r[2]/r[1] ])
        x1,y1 = map(int, [c, -(r[2]+r[0]*c)/r[1] ])
        img1 = cv.line(img1, (x0,y0), (x1,y1), color,1)
        img1 = cv.circle(img1,tuple(pt1),5,color,-1)
        img2 = cv.circle(img2,tuple(pt2),5,color,-1)
    return img1,img2

img5,img6 = drawlines(img1,img2,l1,pts1,pts2)
img3,img4 = drawlines(img2,img1,l2,pts2,pts1)

plt.subplot(121),plt.imshow(img5)
plt.subplot(122),plt.imshow(img3)
plt.show()