import sys
import os
import numpy as np
import cv2
import scipy.signal

# print "hello"
# imgFile = cv.imread('testwgreen.png')

# cv.imshow('image', imgFile)
# cv.waitKey(0)
# cv.destroyAllWindows()


img = cv2.imread('testwgreen.png')
img = img[:,:,1]
img = cv2.medianBlur(img,5)

cimg = cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)
# cimg = img[:,:,1]


circles = cv2.HoughCircles(img,cv2.HOUGH_GRADIENT,1,20,
                            param1=50,param2=30,minRadius=0,maxRadius=50)

circles = np.uint16(np.around(circles))
for i in circles[0,:]:
    # draw the outer circle
    cv2.circle(cimg,(i[0],i[1]),i[2],(0,255,0),2)
    # draw the center of the circle
    cv2.circle(cimg,(i[0],i[1]),2,(0,0,255),3)

cv2.imshow('detected circles',cimg)
cv2.waitKey(0)
cv2.destroyAllWindows()


# cv2.imshow('image', greenimage)
# cv2.waitKey(0)
# cv2.destroyAllWindows()