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

# create sparse point view matrix 
def create_PVM():
    # collecting features
    sift = cv.SIFT_create()
    features = {}
    for i in range(1, 50):
        img = cv.imread(f"Data/House/House/frame000000{i:02d}.png",0)
        features[i] = {}
        features[i]['kp'], features[i]['des'] = sift.detectAndCompute(img, None)

    # collecting matched indices
    matched_idx = {}
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks=50)   # or pass empty dictionary
    flann = cv.FlannBasedMatcher(index_params,search_params)

    fundamental = {}

    # loop over all images/views and return good matches
    for i in range(1, 50):
        if i!=49:
            matches = flann.knnMatch(features[i]['des'],features[i+1]['des'],k=2)

            good = []
            for j,(m,n) in enumerate(matches):
                if m.distance < 0.5*n.distance:
                    # this is pretty crucial, otherwise reconstruction is flat with weird side points. 
                    # the check m.distance < n.distance measures distance between descriptor vectors, whereas 
                    # the distance between two matched keypoints can be checked as follows: 
                    # since the images don't really change much in consecutive views, the actual distance between kp1 and kp2
                    # can't be large. A window in view1 is only slightly moved in view2
               
                    if np.sum(np.abs(np.array(features[i]['kp'][m.queryIdx].pt) - np.array(features[i+1]['kp'][m.trainIdx].pt))) < 4:
                            good.append(m)

            matched_idx[(i,i+1)] = {}
            matched_idx[(i,i+1)][i] = [ m.queryIdx for m in good ]#list(A[ransac_indices])#[ m.queryIdx for m in good ] # reference image indices
            matched_idx[(i,i+1)][i+1] = [ m.trainIdx for m in good ]#list(B[ransac_indices])#[ m.trainIdx for m in good ] # matching image indices
        elif i == 49:
            matches = flann.knnMatch(features[i]['des'],features[1]['des'],k=2)

            good = []
            for j,(m,n) in enumerate(matches):
                if m.distance < 0.5*n.distance:
                    good.append(m)

            A = [ m.queryIdx for m in good]
            B = [ m.trainIdx for m in good]
            try: 

                                matched_idx[(i,1)] = {}
                                matched_idx[(i,1)][i] = A
                                matched_idx[(i,1)][1] = B
            except:
                                
                                matched_idx[(i,1)] = {}
                                matched_idx[(i,1)][i] = A
                                matched_idx[(i,1)][1] = B  


    # row 1,2 - image 1
    new_indicies = matched_idx[(1,2)][1]
    pvm = np.vstack( [ [features[1]['kp'][idx].pt[0] for idx in new_indicies], #x
                       [features[1]['kp'][idx].pt[1] for idx in new_indicies]  #y
                     ] )
    matrix_register = np.array(new_indicies) # [image, image indices in order of columns of pvm]
    matrix_register = matrix_register[np.newaxis,:] #reshaping

    # till row 96 - till image 48
    for i in range(2,49):
        # adding matches from i-1,i images
        mr_new_row = [matched_idx[(i-1,i)][i][matched_idx[(i-1,i)][i-1].index(x)] 
                      if x in matched_idx[(i-1,i)][i-1] 
                      else np.nan 
                      for x in matrix_register[i-2,:]]
        pvm_new_row_x = [features[i]['kp'][int(idx)].pt[0] 
                         if idx == idx
                         else np.nan 
                         for idx in mr_new_row]
        pvm_new_row_y = [features[i]['kp'][int(idx)].pt[1] 
                         if idx == idx
                         else np.nan 
                         for idx in mr_new_row]

        matrix_register = np.vstack([matrix_register, mr_new_row])
        pvm = np.vstack([pvm, pvm_new_row_x, pvm_new_row_y])

        # adding matches from i,i+1 images
        new_indices = list(set(matched_idx[(i,i+1)][i]) - set(matched_idx[(i-1,i)][i]))
        pvm_new_columns = np.vstack( [ np.full([2*(i-1), len(new_indices)], np.nan), 
                                       [features[i]['kp'][idx].pt[0] for idx in new_indices],
                                       [features[i]['kp'][idx].pt[1] for idx in new_indices]
                                     ] )
        pvm = np.append(pvm, pvm_new_columns, axis=1)

        mr_new_columns = np.vstack( [ np.full([(i-1), len(new_indices)], np.nan), new_indices ] )
        matrix_register = np.append(matrix_register, mr_new_columns, axis=1)


    #adding matches from 48,49
    mr_new_row = [matched_idx[(48,49)][49][matched_idx[(48,49)][48].index(x)] 
                  if x in matched_idx[(48,49)][48] 
                  else np.nan 
                  for x in matrix_register[48-1,:]]
    pvm_new_row_x = [features[49]['kp'][int(idx)].pt[0] 
                     if idx == idx
                     else np.nan 
                     for idx in mr_new_row]
    pvm_new_row_y = [features[49]['kp'][int(idx)].pt[1] 
                     if idx == idx
                     else np.nan 
                     for idx in mr_new_row]

    matrix_register = np.vstack([matrix_register, mr_new_row])
    pvm = np.vstack([pvm, pvm_new_row_x, pvm_new_row_y])

    new_indices_1 = list(set(matched_idx[(49,1)][1]) - set(matched_idx[(1,2)][1]))
    new_indices_49 = [matched_idx[(49,1)][49][matched_idx[(49,1)][1].index(x)] 
                      if x in matched_idx[(49,1)][1]
                      else np.nan 
                      for x in new_indices_1]

    pvm_new_columns = np.vstack( [ [features[1]['kp'][idx].pt[0] for idx in new_indices_1],
                                   [features[1]['kp'][idx].pt[1] for idx in new_indices_1],
                                   np.full([2*(48-1), len(new_indices_1)], np.nan), 
                                   [features[49]['kp'][idx].pt[0] for idx in new_indices_49],
                                   [features[49]['kp'][idx].pt[1] for idx in new_indices_49]
                                 ] )
    pvm = np.append(pvm, pvm_new_columns, axis=1)

    mr_new_columns = np.vstack( [ new_indices_1,
                                  np.full([(48-1), len(new_indices_1)], np.nan), 
                                 new_indices_49
                                ] )
    matrix_register = np.append(matrix_register, mr_new_columns, axis=1)

    pvm[np.isnan(pvm)] = 0
    return pvm, fundamental
