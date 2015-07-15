import sys
import os
import numpy as np
import cv2
import scipy.signal

# cv.imshow('image', imgFile)
# cv.waitKey(0)
# cv.destroyAllWindows()

def order_points(pts):
	# initialzie a list of coordinates that will be ordered
	# such that the first entry in the list is the top-left,
	# the second entry is the top-right, the third is the
	# bottom-right, and the fourth is the bottom-left
	rect = np.zeros((4, 2), dtype = "float32")
 
	# the top-left point will have the smallest sum, whereas
	# the bottom-right point will have the largest sum
	s = pts.sum(axis = 1)
	rect[0] = pts[np.argmin(s)]
	rect[2] = pts[np.argmax(s)]
 
	# now, compute the difference between the points, the
	# top-right point will have the smallest difference,
	# whereas the bottom-left will have the largest difference
	diff = np.diff(pts, axis = 1)
	rect[1] = pts[np.argmin(diff)]
	rect[3] = pts[np.argmax(diff)]
 
	# return the ordered coordinates
	return rect

img = cv2.imread('testgreenAw.png')
img = img[:,:,1]
img = cv2.medianBlur(img,5)

cimg = cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)

circles = cv2.HoughCircles(img,cv2.HOUGH_GRADIENT,1,20,
                            param1=80,param2=30,minRadius=0,maxRadius=40)

circles = np.uint16(np.around(circles))
for i in circles[0,:]:
    # draw the outer circle
    cv2.circle(cimg,(i[0],i[1]),i[2],(0,255,0),2)
    # draw the center of the circle
    cv2.circle(cimg,(i[0],i[1]),2,(0,0,255),3)

print circles

cv2.imshow('detected circles',cimg)
cv2.waitKey(0)
cv2.destroyAllWindows()


# img2 = cv2.imread('Dispatcher.png')
# cv2.imshow('image',img2)

# newFridge = np.zeros((img2.shape[1], img2.shape[0]), dtype=np.int)
# warped_image = cv2.warpPerspective(image_1,combination_matrix,(x_max - x_min, y_max - y_min))

# newWarpedImage = cv2.warpPerspective(img2, M, (img2.shape[1], img2.shape[0]))

# test = [[circles[0][0][0], circles[0][0][1]]]

# for i in range(1, len(circles[0])):
# 	print circles[0][i]
# 	for j in range( 0, len(test)):
# 		if ( circles[0][i][0] * circles[0][i][0] + circles[0][i][1] * circles[0][i][1] > test[j][0] * test[j][0] + test[j][1] * test[j][1]):
# 			test.insert(j, [circles[0][i][0], circles[0][i][1]] )
			# print "h"

# print topCircles
# print circles[0][0]
# print test

# cv2.imshow('image', img2)

# M = cv2.getPerspectiveTransform()

# cv2.imshow('image', greenimage)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

