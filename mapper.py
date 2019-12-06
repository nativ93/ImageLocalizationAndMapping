from mappingutils import *

RESIZE_FACTOR = 3

class Mapper:


    def __init__(self, feature_type="sift"):
        self._maps = []
        self._feature_type = feature_type


    def get_map(self):
        return self._maps[0][0]

    def proccess_frame(self, frame, hint=None):

        #if first frame
        if len(self._maps)==0:
            self._maps.append((frame,np.identity(3)))
            self._current_level = 0
            return self._maps[0][1]

        #if no hint or bad hint
        if hint == None or hint > len(self._maps):
            hint = self._current_level

        ref_image = self._maps[hint][0]

        M, mask = findHomography(resize_image(frame, RESIZE_FACTOR), resize_image(ref_image, RESIZE_FACTOR), self._feature_type)
        if M is None:
            return None
        scaleM = np.float32([[RESIZE_FACTOR,0,0],[0,RESIZE_FACTOR,0],[0,0,1]])
        M = np.matmul(scaleM,np.matmul(M,np.linalg.inv(scaleM)))

        h,w,d = frame.shape
        pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
        totalM = M
        dst = cv.perspectiveTransform(pts,M)
        transformRate = rect_size(dst)/(w*h);

        for i in reversed(range(len(self._maps))):
            totalM = np.matmul(self._maps[i][1],(totalM))

        if transformRate < 0.75:
            self._maps.append((frame,M))
            self._current_level+=1
            print("**************LEVEL UPPPPP!!!!****************")

        return totalM

    def locate_frame():
        pass






