import numpy as np
import cv2 as cv
import copy
from mapper import Mapper
import time
# img1 = cv.imread('images/cat_part.png',0)          # queryImage
# img2 = cv.imread('images/cat.png',0) # trainImage
# print(img1.depth())


# def rect_size(pts):
#     p1 = tuple(pts[0][0])
#     p2 = tuple(pts[1][0])
#     p3 = tuple(pts[2][0])
#     # print("p1: "+str(p1))
#     # print("p2: "+str(p2))
#     # print("p3: "+str(p3))
#     h = math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
#     w = math.sqrt((p3[0] - p2[0])**2 + (p3[1] - p2[1])**2)
#     # print("h is: "+str(h))
#     # print("w is: "+str(w))
#     return h*w

# def plot_in_picture(maps,patch_im,feature_type = "sift"):
#     large_im = maps[-1][0];
#     M, mask = findHomography(patch_im,large_im, feature_type);
#     if M is None:
#         return None,None,None
#     h,w,d = patch_im.shape
#     pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
#     dst = cv.perspectiveTransform(pts,M)
#     transformRate = rect_size(dst)/(w*h);
#     # print(dst)
#     # print("transform rate is: "+str(transformRate))
#     # print(dst)
#     for i in reversed(range(len(maps))):
#         # print(maps[i][1])
#         dst = cv.perspectiveTransform(dst,maps[i][1])

#     # print(dst)
#     cv.waitKey()
#     img4 = copy.deepcopy(maps[0][0])
#     for i in range(4):
#         cv.line(img4,tuple(dst[i][0]),tuple(dst[(i+1)%4][0]),(255,0,0),5)

#     # print(rect_size(dst))
#     # print(w*h*0.6)
#     if transformRate < 0.75:
#         maps.append((patch_im,M))
#         print("**************LEVEL UPPPPP!!!!****************")

#     # cv.imshow("4",img4)
#     # cv.imshow("1",patch_im)
#     # cv.imshow("2",large_im)
#     return img4,M,mask
#     # cv.waitKey(0)

def load_video(file, feature_type = "sift"):
    cap = cv.VideoCapture(file)

    map = Mapper("orb")
    ret, frame = cap.read()
    M = map.proccess_frame(frame)
    
    h,w,d = frame.shape
    out = cv.VideoWriter('images/outpy.avi',\
        cv.VideoWriter_fourcc('M','J','P','G'),\
        cap.get(cv.CAP_PROP_FPS), \
        (w,h))

    failes = 0
    i = 0
    frame2 = 0
    max_t = 0
    while failes < 4:
        s = time.time()
        i+=1
        pframe = frame
        ret, frame = cap.read()
        if ret == False:
            break;

        M = map.proccess_frame(frame)
        if M is None:
            print("failllllll")
            failes+=1
            out.write(pframe)
            continue

        h,w,d = frame.shape
        pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
        dst = cv.perspectiveTransform(pts,M)
        img = copy.deepcopy(map.get_map())
        for line in range(4):
            cv.line(img,tuple(dst[line][0]),tuple(dst[(line+1)%4][0]),(255,0,0),5)

        failes = 0
        print(i)
        out.write(img)
        e = time.time()
        if(e-s > max_t):
            max_t = e-s
        print("pip time is "+str(e-s))
    print("max t is: " + str(max_t))
    cap.release()
    out.release()


























