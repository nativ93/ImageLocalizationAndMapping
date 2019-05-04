import numpy as np
import cv2 as cv
import copy
from matplotlib import pyplot as plt
MIN_MATCH_COUNT = 10
# img1 = cv.imread('images/cat_part.png',0)          # queryImage
# img2 = cv.imread('images/cat.png',0) # trainImage
# print(img1.depth())
def findHomography(img1,img2):
    # Initiate SIFT detector
    sift = cv.xfeatures2d.SIFT_create()
    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(img1,None)
    kp2, des2 = sift.detectAndCompute(img2,None)
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks = 50)
    flann = cv.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1,des2,k=2)
    # store all the good matches as per Lowe's ratio test.
    good = []
    for m,n in matches:
        if m.distance < 0.7*n.distance:
            good.append(m)

    if len(good)>MIN_MATCH_COUNT:
        src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
        dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)
        M, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC,5.0)
        return M, mask
    else:
        print( "Not enough matches are found - {}/{}".format(len(good), MIN_MATCH_COUNT) )
        matchesMask = None
        return None,None

def plot_in_picture(large_im,patch_im):
    M, mask = findHomography(patch_im,large_im);
    if M is None:
        return None,None,None
    h,w,d = patch_im.shape
    pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
    dst = cv.perspectiveTransform(pts,M)

    img4 = copy.deepcopy(large_im)
    for i in range(4):
        cv.line(img4,tuple(dst[i][0]),tuple(dst[(i+1)%4][0]),(255,0,0),5)
    cv.imshow("4",img4)
    cv.imshow("1",patch_im)
    cv.imshow("2",large_im)
    return img4,M,mask
    # cv.waitKey(0)

def load_video(file):
    cap = cv.VideoCapture(file)

    ret, map = cap.read()
    h,w,d = map.shape
    map = cv.resize(map,(int(w/3),int(h/3)))
    # cv.imshow("map",map)

    out = cv.VideoWriter('images/outpy.avi',\
        cv.VideoWriter_fourcc('M','J','P','G'),\
        cap.get(cv.CAP_PROP_FPS), \
        (int(w/3),int(h/3)))

    failes = 0
    i = 0
    frame2 = 0
    while failes < 4:
        i+=1
        pframe = frame2
        ret, frame = cap.read()
        frame = cv.resize(frame,(int(w/3),int(h/3)))
        frame2, M, mask = plot_in_picture(map,frame)
        if frame2 is None:
            print("failllllll")
            failes+=1
            out.write(pframe)
            continue
        failes = 0
        print(i)
        out.write(frame2)

    cap.release()
    out.release()


























