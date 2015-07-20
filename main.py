import sys
import os
import numpy as np
import cv2
import scipy.signal

# cv.imshow('image', imgFile)
# cv.waitKey(0)
# cv.destroyAllWindows()

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
    """ Warps image 1 so it can be blended with image 2 (stitched).

    Follow these steps:
        1. Obtain the corners for image 1 and image 2 using the function you
        wrote above.
        
        2. Transform the perspective of the corners of image 1 by using the
        image_1_corners and the homography to obtain the transformed corners.
        
        Note: Now we know the corners of image 1 and image 2. Out of these 8
        points (the transformed corners of image 1 and the corners of image 2),
        we want to find the minimum x, maximum x, minimum y, and maximum y. We
        will need this when warping the perspective of image 1.

        3. Join the two corner arrays together (the transformed image 1 corners,
        and the image 2 corners) into one array of size (8, 1, 2).

        4. For the first column of this array, find the min and max. This will
        be your minimum and maximum X values. Store into x_min, x_max.

        5. For the second column of this array, find the min and max. This will
        be your minimum and maximum Y values. Store into y_min, y_max.

        6. Create a translation matrix that will shift the image by the required
        x_min and y_min (should be a numpy.ndarray). This looks like this:
            [[1, 0, -1 * x_min],
             [0, 1, -1 * y_min],
             [0, 0, 1]]

        Note: We'd like you to explain the reasoning behind multiplying the
        x_min and y_min by negative 1 in your writeup.

        7. Compute the dot product of your translation matrix and the homography
        in order to obtain the homography matrix with a translation.

        8. Then call cv2.warpPerspective. Pass in image 1, the dot product of
        the matrix computed in step 6 and the passed in homography and a vector
        that will fit both images, since you have the corners and their max and
        min, you can calculate it as (x_max - x_min, y_max - y_min).

        9. To finish, you need to blend both images. We have coded the call to
        the blend function for you.

    Args:
        image_1 (numpy.ndarray): Left image.
        image_2 (numpy.ndarray): Right image.
        homography (numpy.ndarray): 3x3 matrix that represents the homography
                                    from image 1 to image 2.

    Returns:
        output_image (numpy.ndarray): T
    he stitched images.
    """
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

    # print warped_image_corners

    # print warped_image_corners
    # print
    # print image_2_corners
    # print


    for i in range( 0, len( warped_image_corners )):
        x_min = min(x_min, warped_image_corners[i][0][0], image_2_corners[i][0][0] )
        y_min = min(y_min, warped_image_corners[i][0][1], image_2_corners[i][0][1] )
        x_max = max(x_max, warped_image_corners[i][0][0], image_2_corners[i][0][0] )
        y_max = max(y_max, warped_image_corners[i][0][1], image_2_corners[i][0][1] )

    # print x_min
    # print x_max
    # print y_min
    # print y_max


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

    # output_image = blendImagePair(warped_image, image_2,
    #                               (-1 * x_min, -1 * y_min))

    # xcheck = output_image.shape[1] - 1
    # while(np.amax(output_image[:,xcheck,0]) == 0 and np.amax(output_image[:,xcheck,1]) == 0 and np.amax(output_image[:,xcheck,2]) == 0 ):
    #     xcheck = xcheck - 1

    # ycheck = output_image.shape[0] - 1
    # while(np.amax(output_image[ycheck,:,0]) == 0 and np.amax(output_image[ycheck,:,1]) == 0 and np.amax(output_image[ycheck,:,2]) == 0 ):
    #     ycheck = ycheck - 1

    # output_image = output_image[0:ycheck + 1,0:xcheck + 1]

    # END OF CODING

    return warped_image

def findHomography(image_1_kp, image_2_kp):
    """ Returns the homography between the keypoints of image 1, image 2, and
        its matches.

    Follow these steps:
        1. Iterate through matches and:
            1a. Get the x, y location of the keypoint for each match. Look up
                the documentation for cv2.DMatch. Image 1 is your query image,
                and Image 2 is your train image. Therefore, to find the correct
                x, y location, you index into image_1_kp using match.queryIdx,
                and index into image_2_kp using match.trainIdx. The x, y point
                is stored in each keypoint (look up documentation).
            1b. Set the keypoint 'pt' to image_1_points and image_2_points, it
                should look similar to this inside your loop:
                    image_1_points[match_idx] = image_1_kp[match.queryIdx].pt
                    # Do the same for image_2 points.

        2. Call cv2.findHomography and pass in image_1_points, image_2_points,
           use method=cv2.RANSAC and ransacReprojThreshold=5.0. I recommend
           you look up the documentation on cv2.findHomography to better
           understand what these parameters mean.
        3. cv2.findHomography returns two values, the homography and a mask.
           Ignore the mask, and simply return the homography.

    Note: 
        The unit test for this function in the included testing script may have 
        value differences and thus may not pass. Please check your image results 
        visually. If your output warped image looks fine, don't worry about this 
        test too much.

    Args:
        image_1_kp (list): The image_1 keypoints, the elements are of type
                           cv2.KeyPoint.
        image_2_kp (list): The image_2 keypoints, the elements are of type 
                           cv2.KeyPoint.
        matches (list): A list of matches. Each item in the list is of type
                        cv2.DMatch.
    Returns:
        homography (numpy.ndarray): A 3x3 homography matrix. Each item in
                                    the matrix is of type numpy.float64.
    """
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

img = cv2.imread('bean.png')
# img = img[:,:,1]
# img = cv2.medianBlur(img,5)

# cimg = cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)

# circles = cv2.HoughCircles(img,cv2.HOUGH_GRADIENT,1,20,
#                             param1=80,param2=30,minRadius=0,maxRadius=40)

# circles = np.uint16(np.around(circles))
# for i in circles[0,:]:
#     # draw the outer circle
#     cv2.circle(cimg,(i[0],i[1]),i[2],(0,255,0),2)
#     # draw the center of the circle
#     cv2.circle(cimg,(i[0],i[1]),2,(0,0,255),3)

# print circles

# cv2.imshow('detected circles',cimg)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


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
    image_1_kp, swapImage, image_2_kp = createData(1, 2, swapImage, points)


    for chunk in range(0, xChunks * yChunks):
        H = findHomography(image_2_kp[chunk], image_1_kp[chunk])
        finalImage = cv2.warpPerspective(swapImage[chunk], H, (originalImage.shape[1], originalImage.shape[0]))
        displayImage = addNewChunk(displayImage, finalImage)
    return displayImage




points = [[123, 340], [194, 299], [165, 385], [238, 359], [178, 443], [258, 445]]
img2 = cv2.imread('Dispatcher.png')
colorOriginal = cv2.imread('bean.png')

colorOriginal = dropOffFinalImage(1,2,colorOriginal, img, img2, points)

cv2.imwrite( 'final.png', colorOriginal)

cv2.imshow('detected circles',colorOriginal)
cv2.waitKey(0)
cv2.destroyAllWindows()


