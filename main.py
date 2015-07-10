import sys
import os
import numpy as np
import cv2 as cv
import scipy.signal

print "hello"
imgFile = cv.imread('1.jpg')

cv.imshow('image', imgFile)
cv.waitKey(0)
cv.destroyAllWindows()