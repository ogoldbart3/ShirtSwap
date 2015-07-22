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
            if ( warpedImage[y, x][0] != 0 ) or ( warpedImage[y, x][1] != 0 ) or ( warpedImage[y, x][2] != 0 ):
                originalImage[y,x] = warpedImage[y,x]

    return originalImage


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

def createData(xChunks, yChunks, imageToBreak, points):
    image_1_kp = []
    # for i in range(0, xChunks * yChunks):
    #     image_1_kp.append([points[i + i / xChunks],points[i + i / xChunks + 1],points[i + i / xChunks + xChunks + 1],points[i + i / xChunks + xChunks + 2]])

    for y in range(0, yChunks):
        for x in range(0, xChunks):
            image_1_kp.append([ points[y*(xChunks+1) + x], points[y*(xChunks+1) + x + 1], points[(y+1)*(xChunks+1) + x], points[(y + 1)*(xChunks+1) + x + 1]])

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
        image[y-2:y+2,x-2:x+2] = [0,255,0]
 
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

colorOriginal = dropOffFinalImage(x,y,colorOriginal, image, swap, points)

cv2.imwrite( 'final.png', colorOriginal)

cv2.imshow('detected circles',colorOriginal)
cv2.waitKey(0)
cv2.destroyAllWindows()