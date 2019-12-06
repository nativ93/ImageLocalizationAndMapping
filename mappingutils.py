import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
import time
MIN_MATCH_COUNT = 10
import math




def getFeatures2d(feature_type):
    print(feature_type)
    if feature_type == "sift":
        return cv.xfeatures2d.SIFT_create()
    elif feature_type == "orb":
        return cv.ORB_create()

def findHomography(img1,img2, feature_type = "sift"):
    # Initiate SIFT detector
    features = getFeatures2d(feature_type)


    # find the keypoints and descriptors with selected feature tpye
    s = time.time()
    kp1, des1 = features.detectAndCompute(img1,None)
    kp2, des2 = features.detectAndCompute(img2,None)
    e = time.time()
    print("ff time is "+str(e-s))

    if des1 is None or des2 is None or len(des1)<5 or len(des2)<5:
    	print("****************BAD FRAME*****************")
    	return None,None

    FLANN_INDEX_KDTREE = 1
    s = time.time()
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks = 50)
    flann = cv.FlannBasedMatcher(index_params, search_params)
    # flann = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)

    matches = flann.knnMatch(np.float32(des1),np.float32(des2),k=2)
    # matches = flann.match(des1,des2)
    e = time.time()
    print("fmatch time is "+str(e-s))
    # store all the good matches as per Lowe's ratio test.
    good = []
    # good = matches
    for m,n in matches:
        if m.distance < 0.7*n.distance:
            good.append(m)

    if len(good)>MIN_MATCH_COUNT:
        s = time.time()
        src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
        dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)
        M, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC,5.0)
        e = time.time()
        print("fh time is "+str(e-s))
        return M, mask
    else:
        print( "Not enough matches are found - {}/{}".format(len(good), MIN_MATCH_COUNT) )
        matchesMask = None
        return None,None

def rect_size(pts):
    p1 = tuple(pts[0][0])
    p2 = tuple(pts[1][0])
    p3 = tuple(pts[2][0])
    h = math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
    w = math.sqrt((p3[0] - p2[0])**2 + (p3[1] - p2[1])**2)
    return h*w











