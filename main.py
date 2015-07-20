import sys
import os
import numpy as np
import cv2
import scipy.signal
import argparse


points = []

def addNewChunk(originalImage, warpedImage):
    for y in range(0, originalImage.shape[0]):
        for x in range(0, originalImage.shape[1]):
            if ( warpedImage[y, x][0] != 0 ) and ( warpedImage[y, x][1] != 0 ) and ( warpedImage[y, x][2] != 0 ):
                originalImage[y,x] = warpedImage[y,x]
    return originalImage

def getImageCorners(image):
    corners = np.zeros((4, 1, 2), dtype=np.float32)
    # WRITE YOUR CODE HERE

    corners[0][0] = [0,0]
    corners[1][0] = [0,image.shape[0]]
    corners[2][0] = [image.shape[1],0]
    corners[3][0] = [image.shape[1],image.shape[0]]

    return corners

def warpImagePair(image_1, image_2, homography):
    
    # Store the result of cv2.warpPerspective in this variable.
    warped_image = None
    # The minimum and maximum values of your corners.
    x_min = 0
    y_min = 0
    x_max = 0
    y_max = 0

    # WRITE YOUR CODE HERE

    image_1_corners = getImageCorners(image_1)
    image_2_corners = getImageCorners(image_2)

    warped_image_corners = np.zeros((4, 1, 2), dtype=np.float32)

    for i in range(0, len(image_1_corners)):
        x = image_1_corners[i][0][0]
        y = image_1_corners[i][0][1]
        warped_image_corners[i][0] = [(homography[0][0] * x + homography[0][1] * y + homography[0][2] )/(homography[2][0]+homography[2][1]+1), (homography[1][0] * x + homography[1][1] * y + homography[1][2] )/(homography[2][0]+homography[2][1]+1)]

    for i in range( 0, len( warped_image_corners )):
        x_min = min(x_min, warped_image_corners[i][0][0], image_2_corners[i][0][0] )
        y_min = min(y_min, warped_image_corners[i][0][1], image_2_corners[i][0][1] )
        x_max = max(x_max, warped_image_corners[i][0][0], image_2_corners[i][0][0] )
        y_max = max(y_max, warped_image_corners[i][0][1], image_2_corners[i][0][1] )

    all_corners = np.zeros((8,1,2), dtype=np.float32)

    for i in range(0,len(warped_image_corners)):
        all_corners[i] = warped_image_corners[i]
        all_corners[i+4] = image_2_corners[i]

    translation_array = np.zeros((3, 3), dtype = np.float32 )
    translation_array[0] = [ 1, 0, -1 * x_min ]
    translation_array[1] = [ 0, 1, -1 * y_min ]
    translation_array[2] = [ 0, 0, 1 ]

    combination_matrix = np.dot( translation_array, homography )

    warped_image = cv2.warpPerspective(image_1,combination_matrix,(x_max - x_min, y_max - y_min))

    return warped_image

def findHomography(image_1_kp, image_2_kp):

    image_1_points = np.zeros((len(image_1_kp), 1, 2), dtype=np.float32)
    image_2_points = np.zeros((len(image_2_kp), 1, 2), dtype=np.float32)

    # WRITE YOUR CODE HERE.

    for i in range(0, len(image_1_kp)):
        image_1_points[i] = [ image_1_kp[i][0],image_1_kp[i][1]]
        image_2_points[i] = [ image_2_kp[i][0],image_2_kp[i][1]]

    output = cv2.findHomography(image_1_points, image_2_points, cv2.RANSAC, 5.0 )

    # Replace this return statement with the homography.
    return output[0]
    # END OF FUNCTION

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

def createData(xChunks, yChunks, imageToBreak, points):
    image_1_kp = []
    for i in range(0, xChunks * yChunks):
        image_1_kp.append([points[i + i / xChunks],points[i + i / xChunks + 1],points[i + i / xChunks + xChunks + 1],points[i + i / xChunks + xChunks + 2]])

    yLen = imageToBreak.shape[0]
    xLen = imageToBreak.shape[1]

    imageBroken = []
    for yChunk in range(0, yLen, yLen / yChunks):
        for xChunk in range(0, xLen, xLen / xChunks):
            imageBroken.append(imageToBreak[yChunk:yChunk + yLen / yChunks, xChunk:xChunk + xLen / xChunks])

    image_2_kp = []
    for i in range(0, xChunks * yChunks):
        image_2_kp.append([[0, 0], [imageBroken[i].shape[1],0], [0, imageBroken[i].shape[0]], [imageBroken[i].shape[1], imageBroken[i].shape[0]]])

    return image_1_kp, imageBroken, image_2_kp


def dropOffFinalImage( xChunks, yChunks, displayImage, originalImage, swapImage, points):
    image_1_kp, swapImage, image_2_kp = createData(xChunks, yChunks, swapImage, points)

    for chunk in range(0, xChunks * yChunks):
        H = findHomography(image_2_kp[chunk], image_1_kp[chunk])
        finalImage = cv2.warpPerspective(swapImage[chunk], H, (originalImage.shape[1], originalImage.shape[0]))
        displayImage = addNewChunk(displayImage, finalImage)
    return displayImage

def click(event, x, y, flags, param):
    # grab references to the global variables
    global points
    if event == cv2.EVENT_LBUTTONUP:
        points.append([x, y])
 
        print [x, y]

# python main.py -i good3.png -s jagface.png -x 2 -y 3

ap = argparse.ArgumentParser()
ap.add_argument("-s", "--swap", required=True, help="Path to the image")
ap.add_argument("-i", "--image", required=True, help="Path to the image")
ap.add_argument("-x", "--xchunks", required=True, help="t")
ap.add_argument("-y", "--ychunks", required=True, help="t")
args = vars(ap.parse_args())
 
# load the image, clone it, and setup the mouse callback function
image = cv2.imread(args["image"])
swap = cv2.imread(args["swap"])
colorOriginal = cv2.imread(args["image"])
x = int(args["xchunks"])
y = int(args["ychunks"])


cv2.namedWindow("image")
cv2.setMouseCallback("image", click)



while len(points) < (x + 1) * (y + 1):
    # display the image and wait for a keypress
    cv2.imshow("image", image)
    key = cv2.waitKey(1) & 0xFF
 
    # if the 'r' key is pressed, reset the cropping region
    if key == ord("r"):
        image = clone.copy()
 
    # if the 'c' key is pressed, break from the loop
    elif key == ord("c"):
        break

cv2.destroyAllWindows()

# pointsForGood3 = [[116,293], [139,258], [148,228], [146,320], [168,286], [191,254], [166,337], [193,313], [214,286], [168,371], [202,359], [228,330]]

colorOriginal = dropOffFinalImage(x,y,colorOriginal, image, swap, points)

cv2.imwrite( 'final.png', colorOriginal)

cv2.imshow('detected circles',colorOriginal)
cv2.waitKey(0)
cv2.destroyAllWindows()


