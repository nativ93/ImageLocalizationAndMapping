import numpy as np
import cv2 as cv
import copy
import math
from matplotlib import pyplot as plt
import time
MIN_MATCH_COUNT = 10
# img1 = cv.imread('images/cat_part.png',0)          # queryImage
# img2 = cv.imread('images/cat.png',0) # trainImage
# print(img1.depth())

def getFeatures2d(feature_type):
    print(feature_type)
    if feature_type == "sift":
        return cv.xfeatures2d.SIFT_create()
    elif feature_type == "orb":
        return cv.ORB_create()

def findHomography(img1,img2, feature_type = "sift"):
    # Initiate SIFT detector
    features = getFeatures2d(feature_type)
    # find the keypoints and descriptors with SIFT
    s = time.time()
    kp1, des1 = features.detectAndCompute(img1,None)
    kp2, des2 = features.detectAndCompute(img2,None)
    e = time.time()
    print("ff time is "+str(e-s))
    FLANN_INDEX_KDTREE = 1
    s = time.time()
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks = 50)
    # flann = cv.FlannBasedMatcher(index_params, search_params)
    flann = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)
    # print()
    # matches = flann.knnMatch(des1,des2,k=2)
    matches = flann.match(des1,des2)
    e = time.time()
    print("fmatch time is "+str(e-s))
    # store all the good matches as per Lowe's ratio test.
    # good = []
    good = matches
    # for m,n in matches:
    #     if m.distance < 0.7*n.distance:
    #         good.append(m)

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
    # print("p1: "+str(p1))
    # print("p2: "+str(p2))
    # print("p3: "+str(p3))
    h = math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
    w = math.sqrt((p3[0] - p2[0])**2 + (p3[1] - p2[1])**2)
    # print("h is: "+str(h))
    # print("w is: "+str(w))
    return h*w

def plot_in_picture(maps,patch_im,feature_type = "sift"):
    large_im = maps[-1][0];
    M, mask = findHomography(patch_im,large_im, feature_type);
    if M is None:
        return None,None,None
    h,w,d = patch_im.shape
    pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
    dst = cv.perspectiveTransform(pts,M)
    transformRate = rect_size(dst)/(w*h);
    # print(dst)
    # print("transform rate is: "+str(transformRate))
    # print(dst)
    for i in reversed(range(len(maps))):
        # print(maps[i][1])
        dst = cv.perspectiveTransform(dst,maps[i][1])

    # print(dst)
    cv.waitKey()
    img4 = copy.deepcopy(maps[0][0])
    for i in range(4):
        cv.line(img4,tuple(dst[i][0]),tuple(dst[(i+1)%4][0]),(255,0,0),5)

    # print(rect_size(dst))
    # print(w*h*0.6)
    if transformRate < 0.75:
        maps.append((patch_im,M))
        print("**************LEVEL UPPPPP!!!!****************")

    # cv.imshow("4",img4)
    # cv.imshow("1",patch_im)
    # cv.imshow("2",large_im)
    return img4,M,mask
    # cv.waitKey(0)

def load_video(file, feature_type = "sift"):
    cap = cv.VideoCapture(file)

    maps = []
    ret, map = cap.read()
    h,w,d = map.shape
    map = cv.resize(map,(int(w/3),int(h/3)))
    maps.append((map,np.identity(3)))
    # cv.imshow("map",map)

    out = cv.VideoWriter('images/outpy.avi',\
        cv.VideoWriter_fourcc('M','J','P','G'),\
        cap.get(cv.CAP_PROP_FPS), \
        (int(w/3),int(h/3)))

    failes = 0
    i = 0
    frame2 = 0
    max_t = 0
    while failes < 4:
        s = time.time()
        i+=1
        pframe = frame2
        ret, frame = cap.read()
        if ret == False:
            break;
        frame = cv.resize(frame,(int(w/3),int(h/3)))
        
        frame2, M, mask = plot_in_picture(maps,frame, feature_type)
        if frame2 is None:
            print("failllllll")
            failes+=1
            out.write(pframe)
            continue
        failes = 0
        print(i)
        out.write(frame2)
        e = time.time()
        if(e-s > max_t):
            max_t = e-s
        print("pip time is "+str(e-s))
    print("max t is: " + str(max_t))
    cap.release()
    out.release()


























