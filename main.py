import findHomography as fh
import cv2 as cv
import numpy as np

def main():

    img1 = cv.imread('images/cat_part.png',0)          # queryImage
    img2 = cv.imread('images/cat.png',0) # trainImage
    # fh.plot_in_picture(img2,img1)
    fh.load_video("images/vid2.mp4")
    cv.waitKey(0)


if __name__ == "__main__":
    main()

